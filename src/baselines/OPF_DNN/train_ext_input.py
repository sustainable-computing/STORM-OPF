import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import util

ALL_GROUPS = [
    "in_distribution", "geo_shift", "extreme_weather",
    "load_extreme_high", "load_extreme_low",
    "fuel_price_up", "fuel_price_down",
    "nk_contingencies", "islanded_operation", "reactive_scarcity",
]
TARGET_GROUPS = ["geo_shift", "extreme_weather", "load_extreme_high", "load_extreme_low"]
_SEED = 0


def _parse_groups(s):
    if s == "all":    return list(ALL_GROUPS)
    if s == "target": return ["in_distribution"] + TARGET_GROUPS
    return [g.strip() for g in s.split(",") if g.strip()]


def _groups_tag(g):
    if sorted(g) == sorted(ALL_GROUPS): return "all"
    if sorted(g) == sorted(["in_distribution"] + TARGET_GROUPS): return "target"
    if g == ["in_distribution"]: return "in_dist_only"
    return "_".join(sorted(g))[:60]


def _load_paths(case, group):
    d = ("src/scenarios/{}/training".format(case) if group == "in_distribution"
         else "src/scenarios/{}/training_stressed/{}".format(case, group))
    mf = pd.read_csv(os.path.join(d, "manifest.csv"))
    return [os.path.join(d, f) for f in mf[mf["status"] == "ok"]["file"].tolist()]


def _bus_load(net, col):
    v = np.zeros(len(net.bus), dtype=float)
    for _, row in net.load.iterrows():
        v[int(row["bus"])] += float(row[col])
    return v


def _make_decoder(enc_dim, branch_out):
    h1 = 2 * enc_dim
    h2 = 2 * branch_out
    return nn.Sequential(
        nn.Linear(enc_dim, h1), nn.ReLU(),
        nn.Linear(h1, h1),      nn.ReLU(),
        nn.Linear(h1, h2),      nn.ReLU(),
        nn.Linear(h2, branch_out),
    )


class OPFDNNExtInput(nn.Module):
    def __init__(self, in_dim, n_gen, n_pv):
        super().__init__()
        enc = 2 * in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, enc), nn.ReLU(),
            nn.Linear(enc, enc),    nn.ReLU(),
        )
        self.dec_pg = _make_decoder(enc, n_gen)
        self.dec_vm = _make_decoder(enc, 1 + n_pv)

    def forward(self, x):
        enc = self.encoder(x)
        return torch.sigmoid(torch.cat([self.dec_pg(enc), self.dec_vm(enc)], dim=-1))


def _build_meta(case):
    base_net, _ = util.get_base_net(case, "in_distribution")
    slack = int(base_net.ext_grid.bus.iloc[0])
    active_gen_idx = [
        i for i, (m, b) in enumerate(zip(
            base_net.gen.in_service.values.astype(bool), base_net.gen.bus.values))
        if m and int(b) != slack
    ]
    pv_bus_list = [int(b) for b in base_net.gen.bus.values if int(b) != slack]
    vm_min = base_net.bus.get("min_vm_pu", pd.Series(0.95, index=base_net.bus.index)).values.astype(float)
    vm_max = base_net.bus.get("max_vm_pu", pd.Series(1.05, index=base_net.bus.index)).values.astype(float)
    pmin = np.array([float(base_net.gen.at[gi, "min_p_mw"]) if "min_p_mw" in base_net.gen.columns else 0.0
                     for gi in active_gen_idx])
    pmax = np.array([float(base_net.gen.at[gi, "max_p_mw"]) if "max_p_mw" in base_net.gen.columns else 1.0
                     for gi in active_gen_idx])
    p_max_nom = pmax.copy()
    c1_nom = np.zeros(len(active_gen_idx))
    c2_nom = np.zeros(len(active_gen_idx))
    if hasattr(base_net, "poly_cost") and len(base_net.poly_cost):
        for k, gi in enumerate(active_gen_idx):
            mask = (base_net.poly_cost["element"] == gi) & (base_net.poly_cost["et"] == "gen")
            if mask.any():
                row = base_net.poly_cost[mask].iloc[0]
                c1_nom[k] = float(row.get("cp1_eur_per_mw", 0.0))
                c2_nom[k] = float(row.get("cp2_eur_per_mw2", 0.0))
    trafo_idx = list(base_net.trafo.index)
    trafo_nom = np.array([float(base_net.trafo.at[t, "max_loading_percent"]) for t in trafo_idx])

    paths = _load_paths(case, "in_distribution")[:2000]
    P_list, Q_list = [], []
    for p in paths:
        net = util.load_net(p)
        P_list.append(_bus_load(net, "p_mw"))
        Q_list.append(_bus_load(net, "q_mvar"))
    P_arr = np.vstack(P_list)
    Q_arr = np.vstack(Q_list)
    P_mu, P_sig = P_arr.mean(0), P_arr.std(0) + 1e-8
    Q_mu, Q_sig = Q_arr.mean(0), Q_arr.std(0) + 1e-8
    nominal_total = float(P_arr.mean())

    return dict(
        slack=slack, active_gen_idx=active_gen_idx, pv_bus_list=pv_bus_list,
        PG_lo=pmin, PG_hi=pmax, Slack_lo=float(vm_min[slack]), Slack_hi=float(vm_max[slack]),
        Vm_lo=vm_min[pv_bus_list], Vm_hi=vm_max[pv_bus_list],
        P_mu=P_mu, P_sig=P_sig, Q_mu=Q_mu, Q_sig=Q_sig,
        p_max_nom=p_max_nom, c1_nom=c1_nom, c2_nom=c2_nom,
        trafo_idx=trafo_idx, trafo_nom=trafo_nom, nominal_total=nominal_total,
    )


def _build_samples(paths, meta):
    eps = 1e-8
    ai = meta["active_gen_idx"]
    pv = meta["pv_bus_list"]
    sl = meta["slack"]
    Xs, Ys = [], []
    for path in paths:
        try:
            net = util.load_net(path)
            P_D = _bus_load(net, "p_mw")
            Q_D = _bus_load(net, "q_mvar")
            total_p = P_D.sum()
            xp = (P_D - meta["P_mu"]) / meta["P_sig"]
            xq = (Q_D - meta["Q_mu"]) / meta["Q_sig"]
            p_frac = P_D / (total_p + eps)
            log_scale = np.clip(np.log((total_p + eps) / (meta["nominal_total"] + eps)), -3, 3)
            p_max_now = np.array([float(net.gen.at[gi, "max_p_mw"]) for gi in ai])
            p_max_ratio = np.clip(p_max_now / np.maximum(meta["p_max_nom"], eps), 0, 2)
            c1_now = np.zeros(len(ai))
            c2_now = np.zeros(len(ai))
            if hasattr(net, "poly_cost") and len(net.poly_cost):
                for k, gi in enumerate(ai):
                    mask = (net.poly_cost["element"] == gi) & (net.poly_cost["et"] == "gen")
                    if mask.any():
                        row = net.poly_cost[mask].iloc[0]
                        c1_now[k] = float(row.get("cp1_eur_per_mw", 0.0))
                        c2_now[k] = float(row.get("cp2_eur_per_mw2", 0.0))
            c1_ratio = np.clip(c1_now / np.where(np.abs(meta["c1_nom"]) > eps, meta["c1_nom"], eps), 0, 5)
            c2_ratio = np.clip(c2_now / np.where(np.abs(meta["c2_nom"]) > eps, meta["c2_nom"], eps), 0, 5)
            t_now = np.array([float(net.trafo.at[t, "max_loading_percent"]) for t in meta["trafo_idx"]])
            trafo_ratio = np.clip(t_now / np.maximum(meta["trafo_nom"], eps), 0, 2)
            x = np.concatenate([xp, xq, p_frac, [log_scale], p_max_ratio, c1_ratio, c2_ratio, trafo_ratio])
            x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
            PG = np.array([float(net.res_gen.at[gi, "p_mw"]) for gi in ai])
            sv = float(net.res_bus.at[sl, "vm_pu"])
            vm_pv = net.res_bus.vm_pu.values[pv].astype(float)
            PG_n = (PG - meta["PG_lo"]) / (meta["PG_hi"] - meta["PG_lo"] + eps)
            sv_n = (sv - meta["Slack_lo"]) / (meta["Slack_hi"] - meta["Slack_lo"] + eps)
            vm_n = (vm_pv - meta["Vm_lo"]) / (meta["Vm_hi"] - meta["Vm_lo"] + eps)
            y = np.concatenate([PG_n, [sv_n], vm_n])
            if not np.isfinite(y).all():
                continue
            Xs.append(x)
            Ys.append(y)
        except Exception:
            continue
    return (torch.tensor(np.vstack(Xs), dtype=torch.float32),
            torch.tensor(np.vstack(Ys), dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--groups", default="in_distribution")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    groups = _parse_groups(args.groups)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print("Building meta from {} nominal scenarios...".format(args.case))
    meta = _build_meta(args.case)

    all_X, all_Y = [], []
    for g in groups:
        X, Y = _build_samples(_load_paths(args.case, g), meta)
        print("  {}: {} samples".format(g, len(X)))
        all_X.append(X)
        all_Y.append(Y)
    X = torch.cat(all_X)
    Y = torch.cat(all_Y)
    print("  Total: {} samples  in_dim={}  out_dim={}".format(len(X), X.shape[1], Y.shape[1]))

    in_dim = X.shape[1]
    n_gen = len(meta["active_gen_idx"])
    n_pv = len(meta["pv_bus_list"])
    model = OPFDNNExtInput(in_dim, n_gen, n_pv).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    loader = DataLoader(TensorDataset(X.to(device), Y.to(device)), batch_size=args.batch, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if (epoch + 1) % 10 == 0:
            print("  epoch {:4d}/{}  mse={:.4e}".format(epoch + 1, args.epochs, total / len(loader)))

    tag = _groups_tag(groups)
    model_dir = os.path.join("src", "baselines", "OPF_DNN", "model", args.case)
    os.makedirs(model_dir, exist_ok=True)
    ckpt = os.path.join(model_dir, "ext_input_{}.pth".format(tag))
    torch.save({"state_dict": model.state_dict(), "meta": meta,
                "in_dim": in_dim, "n_gen": n_gen, "n_pv": n_pv,
                "groups": groups, "model_class": "OPFDNNExtInput"}, ckpt)
    print("Saved -> {}".format(ckpt))


if __name__ == "__main__":
    main()
