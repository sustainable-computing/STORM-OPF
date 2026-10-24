import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import global_vars_dc as gvars
from src.global_evaluation_DCSC import DCSCEvaluator

from . import vars
from .train_model import DeepOPFSCDCOPFNet, DEFAULT_HIDDEN


TINY = 1e-8


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _list_test_scenarios(case: str) -> List[str]:
    root = os.path.join("src", "scenarios", case, "test")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    names = []
    for name in sorted(os.listdir(root)):
        scen_dir = os.path.join(root, name)
        manifest_path = os.path.join(scen_dir, "manifest.csv")
        if os.path.isdir(scen_dir) and os.path.exists(manifest_path):
            names.append(name)
    return names


def _read_manifest_paths(scen_dir: str, max_samples: Optional[int] = None) -> List[str]:
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    files = manifest["file"].tolist()
    if max_samples is not None and max_samples > 0:
        files = files[: int(max_samples)]
    return [os.path.join(scen_dir, f) for f in files]


def _collect_test_paths(case: str, which_data: str, max_samples_test: Optional[int] = None) -> List[str]:
    if which_data == "all":
        paths: List[str] = []
        for name in _list_test_scenarios(case):
            scen_dir = os.path.join("src", "scenarios", case, "test", name)
            paths.extend(_read_manifest_paths(scen_dir, max_samples=max_samples_test))
        return paths

    scen_dir = os.path.join("src", "scenarios", case, "test", which_data)
    return _read_manifest_paths(scen_dir, max_samples=max_samples_test)


def _load_nominal_x_stats(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    pd_path = os.path.join(data_dir, "nominal_PD_train.npy")
    PD = np.load(pd_path).astype(np.float32)

    x_mu = PD.mean(axis=0).astype(np.float32)
    x_sd = (PD.std(axis=0) + TINY).astype(np.float32)
    return x_mu, x_sd


def _load_test_loader_and_meta(
    data_dir: str,
    which_data: str,
    batch_size: int,
    x_mu: np.ndarray,
    x_sd: np.ndarray,
):
    pd_path = os.path.join(data_dir, f"PD_test_{which_data}.npy")
    pg_path = os.path.join(data_dir, f"PG_test_{which_data}.npy")
    meta_path = os.path.join(data_dir, f"test_meta_{which_data}.pt")

    X_raw = np.load(pd_path).astype(np.float32)
    Y_raw = np.load(pg_path).astype(np.float32)
    meta = dict(_safe_torch_load(meta_path))

    Xn = ((X_raw - x_mu[None, :]) / x_sd[None, :]).astype(np.float32)

    pmin_pu = np.asarray(meta["pmin_pu"], dtype=np.float32)
    pmax_pu = np.asarray(meta["pmax_pu"], dtype=np.float32)
    Yn = ((Y_raw - pmin_pu[None, :]) / (pmax_pu - pmin_pu + TINY)[None, :]).astype(np.float32)

    meta["x_mu_pu"] = x_mu.astype(np.float32)
    meta["x_sd_pu"] = x_sd.astype(np.float32)

    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, meta


def _choose_model_path(case: str, which_model: str) -> str:
    model_dir = os.path.join("src", "baselines", "DeepOPF_SCDCOPF", "model", case)
    if which_model == "baseline":
        return os.path.join(model_dir, "trained_model_nominal.pth")
    if which_model == "augmented":
        return os.path.join(model_dir, "trained_model_augmented.pth")
    raise ValueError(f"Unsupported which_model: {which_model}")


def deepopf_scdcopf_denorm_outputs(y_norm: torch.Tensor, meta: Dict, device: str = "cpu"):
    if not torch.is_tensor(y_norm):
        y_norm = torch.as_tensor(y_norm, dtype=torch.float32, device=device)
    pmin_pu = torch.as_tensor(meta["pmin_pu"], dtype=y_norm.dtype, device=device).view(1, -1)
    pmax_pu = torch.as_tensor(meta["pmax_pu"], dtype=y_norm.dtype, device=device).view(1, -1)
    base_mva = float(meta["baseMVA"])
    pg_pu = pmin_pu + (pmax_pu - pmin_pu) * y_norm
    return pg_pu * base_mva


def _make_model_infer_fn(meta: Dict, device: str = "cpu"):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    x_mu_t = torch.as_tensor(meta["x_mu_pu"], dtype=torch.float32, device=dev).view(1, -1)
    x_sd_t = torch.as_tensor(meta["x_sd_pu"], dtype=torch.float32, device=dev).view(1, -1)

    pmin_pu_t = torch.as_tensor(meta["pmin_pu"], dtype=torch.float32, device=dev).view(1, -1)
    pmax_pu_t = torch.as_tensor(meta["pmax_pu"], dtype=torch.float32, device=dev).view(1, -1)

    pred_idx = list(np.asarray(meta["pred_idx"], dtype=int).tolist())
    balance_idx = int(meta["balance_idx"])
    ng = int(meta["ng"])

    def infer_fn(model, xb, device: str = "cpu"):
        xb = xb.to(dev).float()
        alpha = model(xb)

        pd_fullbus_pu = xb * x_sd_t + x_mu_t

        lo = pmin_pu_t[:, pred_idx]
        hi = pmax_pu_t[:, pred_idx]
        pg_nonbal_pu = lo + (hi - lo) * alpha

        B = xb.shape[0]
        pg_full_pu = torch.zeros((B, ng), dtype=xb.dtype, device=dev)
        pg_full_pu[:, pred_idx] = pg_nonbal_pu
        pg_full_pu[:, balance_idx] = pd_fullbus_pu.sum(dim=1) - pg_nonbal_pu.sum(dim=1)

        pg_full_norm = (pg_full_pu - pmin_pu_t) / (pmax_pu_t - pmin_pu_t + TINY)
        return pg_full_norm

    return infer_fn


def deepopf_scdcopf_apply_prediction_to_case(dc_case, meta, x_norm_i, y_norm_i, denorm_fn, device: str = "cpu"):
    y1 = (
        y_norm_i.detach().to(device).view(1, -1)
        if hasattr(y_norm_i, "detach")
        else torch.as_tensor(y_norm_i, dtype=torch.float32, device=device).view(1, -1)
    )
    pg = denorm_fn(y1, meta, device=device).squeeze(0).detach().cpu().numpy().astype(float)
    return {"pg": pg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--which_data", type=str, default="in_distribution")
    parser.add_argument("--which_model", type=str, default="baseline", choices=["baseline", "augmented"])
    parser.add_argument("--max_samples_test", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=vars.batch_size)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--no_projection", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    ds_dir = os.path.join("src", "baselines", "DeepOPF_SCDCOPF", "datasets", args.case, "repo_style")
    os.makedirs(ds_dir, exist_ok=True)


    x_mu, x_sd = _load_nominal_x_stats(ds_dir)

    try:
        loader, meta = _load_test_loader_and_meta(
            data_dir=ds_dir,
            which_data=args.which_data,
            batch_size=args.batch,
            x_mu=x_mu,
            x_sd=x_sd,
        )
        # print("loaded test samples:", len(loader.dataset))
        # print("X shape:", loader.dataset.tensors[0].shape)
        # print("Y shape:", loader.dataset.tensors[1].shape)
        # print("num batches:", len(loader))
    except FileNotFoundError:
        print(f"Missing repo-style test files for {args.which_data}. Generate data first by running make_dataset.")
        sys.exit(1)

    X0, Y0 = next(iter(loader))
    model = DeepOPFSCDCOPFNet(int(X0.shape[1]), args.hidden, int(len(meta["pred_idx"]))).to(device)

    model_path = _choose_model_path(args.case, args.which_model)
    if not os.path.exists(model_path):
        print(f"Missing model file: {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    net_paths = _collect_test_paths(
        case=args.case,
        which_data=args.which_data,
        max_samples_test=args.max_samples_test,
    )

    evaluator = DCSCEvaluator(
        denorm_fn=deepopf_scdcopf_denorm_outputs,
        apply_pred_fn=deepopf_scdcopf_apply_prediction_to_case,
        model_infer_fn=_make_model_infer_fn(meta, device=str(device)),
        feas_tol=getattr(gvars, "dc_balance_tol_mw", 1e-4),
        max_penalty=getattr(gvars, "dc_max_pen", 1e6),
        ctol=getattr(gvars, "dc_ctol", 1e-7),
        project_if_needed=getattr(gvars, "dc_project_if_needed", True),
    )

    res = evaluator.evaluate(
        model=model,
        loader=loader,
        meta=meta,
        net_paths=net_paths,
        device=str(device),
        collect_norm_mse=True,
        verbose=False,
    )

    print(res["feasibility_rate_%"])
    os.makedirs(os.path.join("results", args.case), exist_ok=True)
    proj_tag = "noproject" if args.no_projection else "project"
    out_path = os.path.join(
        "results",
        args.case,
        f"DeepOPF_SCDCOPF_{proj_tag}_{args.which_model}_result_{args.which_data}.pkl",
    )
    with open(out_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved DeepOPF_SCDCOPF results to {out_path}")
