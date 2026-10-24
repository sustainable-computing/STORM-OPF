import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


TINY = 1e-8
DEFAULT_HIDDEN = 8


torch.manual_seed(1)
np.random.seed(1)


class DeepOPFSCDCOPFNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 4 * hidden)
        self.layer2 = nn.Linear(4 * hidden, 2 * hidden)
        self.layer3 = nn.Linear(2 * hidden, hidden)
        self.layer4 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x


class Penalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        hinge = pred * pred - 1.0
        hinge = torch.where(hinge > 0.0, hinge, torch.zeros_like(hinge))
        return torch.mean(hinge)


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _derive_gen_bus(meta: Dict) -> List[int]:
    if "Cg" in meta:
        Cg = np.asarray(meta["Cg"])
        if Cg.ndim == 2:
            return np.argmax(Cg, axis=0).astype(int).tolist()

    if "gen_signatures" in meta:
        sig = meta["gen_signatures"]
        out = []
        for s in sig:
            if isinstance(s, dict) and "bus" in s:
                out.append(int(s["bus"]))
        if out:
            return out

    raise KeyError("Could not derive generator-to-bus mapping from meta['Cg'] or meta['gen_signatures'].")


def _torch_meta(meta: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    out["pmin_pu"] = torch.as_tensor(meta["pmin_pu"], dtype=torch.float32, device=device)
    out["pmax_pu"] = torch.as_tensor(meta["pmax_pu"], dtype=torch.float32, device=device)

    out["balance_idx"] = int(meta["balance_idx"])
    out["pred_idx"] = list(np.asarray(meta["pred_idx"], dtype=int).tolist())
    out["nb"] = int(meta["nb"])
    out["gen_bus"] = _derive_gen_bus(meta)

    base_mva = float(meta["baseMVA"])
    Hnorm_list: List[torch.Tensor] = []
    const_norm_list: List[torch.Tensor] = []

    for H, const, rate_mw in zip(meta["H_list"], meta["const_list"], meta["rate_list_mw"]):
        H = np.asarray(H, dtype=np.float32)
        const = np.asarray(const, dtype=np.float32).reshape(-1)
        rate_mw = np.asarray(rate_mw, dtype=np.float32).reshape(-1)

        rate_safe = rate_mw.copy()
        rate_safe[np.abs(rate_safe) < TINY] = 1.0

        Hnorm = H * (base_mva / rate_safe[:, None])
        const_norm = const / rate_safe

        Hnorm_list.append(torch.as_tensor(Hnorm, dtype=torch.float32, device=device))
        const_norm_list.append(torch.as_tensor(const_norm, dtype=torch.float32, device=device))

    out["Hnorm_list"] = Hnorm_list
    out["const_norm_list"] = const_norm_list
    return out


def _reconstruct_pg_from_alpha(
    pred_alpha: torch.Tensor,
    load_pu_fullbus: torch.Tensor,
    meta_t: Dict[str, torch.Tensor],
) -> torch.Tensor:
    '''
    reconstruction step
    '''
    pred_idx = meta_t["pred_idx"]
    balance_idx = meta_t["balance_idx"]

    pmin = meta_t["pmin_pu"].view(1, -1)
    pmax = meta_t["pmax_pu"].view(1, -1)

    lo = pmin[:, pred_idx]
    hi = pmax[:, pred_idx]
    pg_pred = lo + (hi - lo) * pred_alpha

    B = pred_alpha.shape[0]
    ng = pmin.shape[1]
    pg_full = pred_alpha.new_zeros((B, ng))
    pg_full[:, pred_idx] = pg_pred

    total_load = load_pu_fullbus.sum(dim=1)
    other_sum = pg_pred.sum(dim=1)
    pg_full[:, balance_idx] = total_load - other_sum
    return pg_full


def _normalize_full_pg(pg_full: torch.Tensor, meta_t: Dict[str, torch.Tensor]) -> torch.Tensor:
    pmin = meta_t["pmin_pu"].view(1, -1)
    pmax = meta_t["pmax_pu"].view(1, -1)
    return (pg_full - pmin) / (pmax - pmin + TINY)


def _build_bus_injections(
    pg_full_pu: torch.Tensor,
    load_pu_fullbus: torch.Tensor,
    meta_t: Dict[str, torch.Tensor],
) -> torch.Tensor:
    B = pg_full_pu.shape[0]
    nb = int(meta_t["nb"])
    gen_bus = meta_t["gen_bus"]

    predictionG = pg_full_pu.new_zeros((B, nb))
    for j, b in enumerate(gen_bus):
        predictionG[:, b] += pg_full_pu[:, j]

    predictionD = load_pu_fullbus
    return predictionG - predictionD


def _prepare_targets_full_pg_norm(PG_pu: np.ndarray, meta: Dict) -> np.ndarray:
    pmin_pu = np.asarray(meta["pmin_pu"], dtype=np.float32)
    pmax_pu = np.asarray(meta["pmax_pu"], dtype=np.float32)
    Yn = ((PG_pu - pmin_pu[None, :]) / (pmax_pu - pmin_pu + TINY)[None, :]).astype(np.float32)
    return Yn


def _load_nominal_repo_style_data(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    pd_path = os.path.join(data_dir, "nominal_PD_train.npy")
    pg_path = os.path.join(data_dir, "nominal_PG_train.npy")
    meta_path = os.path.join(data_dir, "training_meta_nominal.pt")

    PD = np.load(pd_path).astype(np.float32)
    PG = np.load(pg_path).astype(np.float32)
    meta = dict(_safe_torch_load(meta_path))

    x_mu = np.mean(PD, axis=0).astype(np.float32)
    x_sd = (np.std(PD, axis=0) + TINY).astype(np.float32)

    meta["x_mu_pu"] = x_mu
    meta["x_sd_pu"] = x_sd

    Xn = ((PD - x_mu) / x_sd).astype(np.float32)
    Yn = _prepare_targets_full_pg_norm(PG, meta)
    return Xn, Yn, meta


def _check_meta_compatible(meta_ref: Dict, meta_other: Dict, name: str) -> None:
    if abs(float(meta_ref["baseMVA"]) - float(meta_other["baseMVA"])) > 1e-8:
        raise ValueError(f"Metadata mismatch for {name} at field 'baseMVA'.")
    for k in ["nb", "ng", "balance_idx"]:
        if int(meta_ref[k]) != int(meta_other[k]):
            raise ValueError(f"Metadata mismatch for {name} at field '{k}'.")
    if list(np.asarray(meta_ref["pred_idx"]).tolist()) != list(np.asarray(meta_other["pred_idx"]).tolist()):
        raise ValueError(f"Metadata mismatch for {name} at field 'pred_idx'.")


def _load_all_stressed_repo_style_data(
    data_dir: str,
    nominal_meta: Dict,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pd_files = sorted(glob.glob(os.path.join(data_dir, "stressed_*_PD_train.npy")))
    if not pd_files:
        raise FileNotFoundError(
            f"No stressed repo-style datasets found in {data_dir}. "
            f"Expected files like stressed_<case>_PD_train.npy"
        )

    Xn_parts: List[np.ndarray] = []
    Yn_parts: List[np.ndarray] = []
    stress_names: List[str] = []

    x_mu = np.asarray(nominal_meta["x_mu_pu"], dtype=np.float32)
    x_sd = np.asarray(nominal_meta["x_sd_pu"], dtype=np.float32)

    for pd_path in pd_files:
        base = os.path.basename(pd_path)
        stress_name = base[len("stressed_") : -len("_PD_train.npy")]
        pg_path = os.path.join(data_dir, f"stressed_{stress_name}_PG_train.npy")
        meta_path = os.path.join(data_dir, f"training_meta_stressed_{stress_name}.pt")

        if not os.path.exists(pg_path):
            raise FileNotFoundError(f"Missing stressed PG file: {pg_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing stressed meta file: {meta_path}")

        PD = np.load(pd_path).astype(np.float32)
        PG = np.load(pg_path).astype(np.float32)
        meta = dict(_safe_torch_load(meta_path))
        _check_meta_compatible(nominal_meta, meta, name=stress_name)

        Xn = ((PD - x_mu) / x_sd).astype(np.float32)
        Yn = _prepare_targets_full_pg_norm(PG, nominal_meta)

        Xn_parts.append(Xn)
        Yn_parts.append(Yn)
        stress_names.append(stress_name)

    Xn_all = np.vstack(Xn_parts).astype(np.float32)
    Yn_all = np.vstack(Yn_parts).astype(np.float32)
    return Xn_all, Yn_all, stress_names


def _make_loader_from_numpy(Xn: np.ndarray, Yn: np.ndarray, batch_size: int, shuffle: bool) -> Data.DataLoader:
    ds = Data.TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn))
    loader = Data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


def _train_one_epoch(
    model: nn.Module,
    loader: Data.DataLoader,
    meta: Dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    penalty: nn.Module,
    dev: torch.device,
) -> Tuple[float, float, float, float]:
    meta_t = _torch_meta(meta, dev)
    x_mu_t = torch.as_tensor(meta["x_mu_pu"], dtype=torch.float32, device=dev).view(1, -1)
    x_sd_t = torch.as_tensor(meta["x_sd_pu"], dtype=torch.float32, device=dev).view(1, -1)

    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_base = 0.0
    running_sc = 0.0
    n_batches = 0

    for train_x, train_y in loader:
        train_x = train_x.to(dev, dtype=torch.float32)
        train_y = train_y.to(dev, dtype=torch.float32)

        pred_alpha = model(train_x)
        train_D_fullbus = train_x * x_sd_t + x_mu_t
        pg_full = _reconstruct_pg_from_alpha(pred_alpha, train_D_fullbus, meta_t)
        pred_full_norm = _normalize_full_pg(pg_full, meta_t)

        train_loss1 = criterion(pred_full_norm, train_y)

        inj = _build_bus_injections(pg_full, train_D_fullbus, meta_t)

        base_trans = inj @ meta_t["Hnorm_list"][0].T + meta_t["const_norm_list"][0].view(1, -1)
        train_loss3 = penalty(base_trans)

        train_loss4 = pg_full.new_tensor(0.0)
        if len(meta_t["Hnorm_list"]) > 1:
            sc_terms: List[torch.Tensor] = []
            for Hn, cn in zip(meta_t["Hnorm_list"][1:], meta_t["const_norm_list"][1:]):
                trans = inj @ Hn.T + cn.view(1, -1)
                sc_terms.append(penalty(trans))
            train_loss4 = torch.stack(sc_terms).mean()

        loss = train_loss1 + train_loss3 + train_loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mse += train_loss1.item()
        running_base += train_loss3.item()
        running_sc += train_loss4.item()
        n_batches += 1

    return (
        running_loss / max(1, n_batches),
        running_mse / max(1, n_batches),
        running_base / max(1, n_batches),
        running_sc / max(1, n_batches),
    )


def train_baseline_model(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    hidden: int
):
    Xn_nom, Yn_nom, meta = _load_nominal_repo_style_data(data_dir=data_dir)

    train_loader = _make_loader_from_numpy(Xn_nom, Yn_nom, batch_size=batch_size, shuffle=True)

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    in_dim = train_loader.dataset[0][0].numel()
    out_dim = len(meta["pred_idx"])

    model = DeepOPFSCDCOPFNet(in_dim, hidden, out_dim).to(dev)
    penalty = Penalty()
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("input dim:", in_dim)
    print("target dim:", train_loader.dataset[0][1].numel())
    print("dict_keys(meta):", meta.keys())
    print("num contingencies including base:", len(meta["H_list"]))
    print(f"baseline nominal samples: {len(train_loader.dataset)}")

    for epoch in range(epochs):
        loss, mse, base_pen, sc_pen = _train_one_epoch(
            model=model,
            loader=train_loader,
            meta=meta,
            optimizer=optimizer,
            criterion=criterion,
            penalty=penalty,
            dev=dev,
        )
        print(
            f"[epoch {epoch + 1:03d}] "
            f"loss={loss:.6e} mse={mse:.6e} "
            f"base_pen={base_pen:.6e} sc_pen={sc_pen:.6e}"
        )

    return model


def train_augmented_model(
    data_dir: str,
    model_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    hidden: int,
    nominal_model_name: str,
    ramp_start: float,
    ramp_end: float,
    ramp_epochs: int,
):
    Xn_nom, Yn_nom, meta = _load_nominal_repo_style_data(
        data_dir=data_dir)
    Xn_stress, Yn_stress, stress_names = _load_all_stressed_repo_style_data(
        data_dir=data_dir,
        nominal_meta=meta,
    )

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    in_dim = Xn_nom.shape[1]
    out_dim = len(meta["pred_idx"])

    model = DeepOPFSCDCOPFNet(in_dim, hidden, out_dim).to(dev)

    nominal_model_path = os.path.join(model_dir, nominal_model_name)
    if not os.path.exists(nominal_model_path):
        raise FileNotFoundError(
            f"Nominal baseline model not found: {nominal_model_path}. "
            f"Train the baseline model first."
        )

    state = torch.load(nominal_model_path, map_location=dev)
    model.load_state_dict(state)
    print(f"[init] loaded nominal weights from {nominal_model_path}")
    print(f"[init] stressed datasets found: {stress_names}")
    print(f"[init] nominal samples={len(Xn_nom)} stressed samples={len(Xn_stress)}")

    penalty = Penalty()
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        if ramp_epochs <= 1:
            stress_frac = ramp_end
        else:
            t = min(epoch, ramp_epochs - 1) / float(ramp_epochs - 1)
            stress_frac = ramp_start + (ramp_end - ramp_start) * t
        stress_frac = float(max(0.0, min(1.0, stress_frac)))

        n_stress_total = Xn_stress.shape[0]
        n_stress_use = int(round(stress_frac * n_stress_total))
        n_stress_use = max(0, min(n_stress_total, n_stress_use))

        if n_stress_use > 0:
            idx = np.random.permutation(n_stress_total)[:n_stress_use]
            X_epoch = np.vstack([Xn_nom, Xn_stress[idx]]).astype(np.float32)
            Y_epoch = np.vstack([Yn_nom, Yn_stress[idx]]).astype(np.float32)
        else:
            X_epoch = Xn_nom
            Y_epoch = Yn_nom

        train_loader = _make_loader_from_numpy(X_epoch, Y_epoch, batch_size=batch_size, shuffle=True)

        loss, mse, base_pen, sc_pen = _train_one_epoch(
            model=model,
            loader=train_loader,
            meta=meta,
            optimizer=optimizer,
            criterion=criterion,
            penalty=penalty,
            dev=dev,
        )

        print(
            f"[epoch {epoch + 1:03d}] "
            f"stress_frac={stress_frac:.4f} "
            f"nominal={len(Xn_nom)} stressed={n_stress_use} total={len(X_epoch)} "
            f"loss={loss:.6e} mse={mse:.6e} "
            f"base_pen={base_pen:.6e} sc_pen={sc_pen:.6e}"
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--which_model", type=str, default="baseline", choices=["baseline", "augmented"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_threads", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--nominal_model_name", type=str, default="trained_model_nominal.pth")
    parser.add_argument("--ramp_start", type=float, default=0.2)
    parser.add_argument("--ramp_end", type=float, default=1.0)
    parser.add_argument("--ramp_epochs", type=int, default=5)

    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_dir = args.data_dir or os.path.join(
        "src", "baselines", "DeepOPF_SCDCOPF", "datasets", args.case, "repo_style"
    )
    model_dir = os.path.join("src", "baselines", "DeepOPF_SCDCOPF", "model", args.case)
    os.makedirs(model_dir, exist_ok=True)

    if args.which_model == "baseline":
        model = train_baseline_model(
            data_dir=data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            hidden=args.hidden
        )
        model_path = os.path.join(model_dir, "trained_model_nominal.pth")
    else:
        model = train_augmented_model(
            data_dir=data_dir,
            model_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            hidden=args.hidden,
            nominal_model_name=args.nominal_model_name,
            ramp_start=args.ramp_start,
            ramp_end=args.ramp_end,
            ramp_epochs=args.ramp_epochs,
        )
        model_path = os.path.join(model_dir, "trained_model_augmented.pth")

    torch.save(model.state_dict(), model_path)
    print(f"[done] saved model to {model_path}")
