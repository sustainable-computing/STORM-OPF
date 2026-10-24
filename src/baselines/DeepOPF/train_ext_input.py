import argparse
import copy
import gzip
import io
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader

from src import util
from src.baselines.DeepOPF.train_model import (
    DeepOPFPenaltyNet,
    build_hist_initialization,
    prep_constraints,
    run_pf,
    compute_penalty_from_net,
)
from src.baselines.DeepOPF import vars

ALL_GROUPS = [
    "in_distribution", "geo_shift", "extreme_weather",
    "load_extreme_high", "load_extreme_low",
    "fuel_price_up", "fuel_price_down",
    "nk_contingencies", "islanded_operation", "reactive_scarcity",
]
TARGET_GROUPS = ["geo_shift", "extreme_weather", "load_extreme_high", "load_extreme_low"]


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


def _fast_load_tables(path, tables):
    """Read only the named DataFrame tables from a pandapower .json.gz file.

    ~36x faster than pp.from_json because it skips network object construction
    and parses only the tables we actually need.
    """
    with gzip.open(path, "rb") as f:
        obj = json.loads(f.read())["_object"]
    return {t: pd.read_json(io.StringIO(obj[t]["_object"]), orient="split") for t in tables}


_SAMPLE_META = None


def _worker_init(meta=None):
    """Keep each build worker single-threaded to avoid CPU oversubscription."""
    global _SAMPLE_META
    _SAMPLE_META = meta
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _mp_context():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


def _load_pq_worker(path):
    tables = _fast_load_tables(path, ["bus", "load"])
    n_bus = len(tables["bus"])
    load_df = tables["load"]
    P_D = np.zeros(n_bus)
    Q_D = np.zeros(n_bus)
    agg = load_df.groupby("bus")[["p_mw", "q_mvar"]].sum()
    P_D[agg.index.values.astype(int)] = agg["p_mw"].values
    Q_D[agg.index.values.astype(int)] = agg["q_mvar"].values
    return P_D, Q_D


def _build_meta(case, build_workers=0):
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

    paths = _load_paths(case, "in_distribution")
    if build_workers > 1:
        chunksize = max(1, len(paths) // (build_workers * 4))
        with _mp_context().Pool(
            processes=build_workers,
            initializer=_worker_init,
        ) as pool:
            rows = pool.map(_load_pq_worker, paths, chunksize=chunksize)
    else:
        rows = [_load_pq_worker(p) for p in paths]

    # pool.map preserves path order, so meta is identical to serial construction.
    P_arr = np.vstack([p for p, _ in rows])
    Q_arr = np.vstack([q for _, q in rows])
    P_mu, P_sig = P_arr.mean(0), P_arr.std(0) + 1e-8
    Q_mu, Q_sig = Q_arr.mean(0), Q_arr.std(0) + 1e-8
    nominal_total = float(P_arr.mean())

    return dict(
        slack_bus=slack, active_gen_idx=active_gen_idx, pv_bus_list=pv_bus_list,
        n_bus=len(base_net.bus),
        PG_lo=pmin, PG_hi=pmax, Slack_lo=float(vm_min[slack]), Slack_hi=float(vm_max[slack]),
        Vm_lo=vm_min[pv_bus_list], Vm_hi=vm_max[pv_bus_list],
        P_mu=P_mu, P_sig=P_sig, Q_mu=Q_mu, Q_sig=Q_sig,
        p_max_nom=p_max_nom, c1_nom=c1_nom, c2_nom=c2_nom,
        trafo_idx=trafo_idx, trafo_nom=trafo_nom, nominal_total=nominal_total,
    )


def _build_one_sample(path, meta):
    eps = 1e-8
    ai = meta["active_gen_idx"]
    pv = meta["pv_bus_list"]
    sl = meta["slack_bus"]
    n_bus = meta["n_bus"]
    try:
        tables = _fast_load_tables(
            path, ["load", "res_gen", "res_bus", "gen", "poly_cost", "trafo"]
        )

        load_df = tables["load"]
        P_D = np.zeros(n_bus)
        Q_D = np.zeros(n_bus)
        agg = load_df.groupby("bus")[["p_mw", "q_mvar"]].sum()
        P_D[agg.index.values.astype(int)] = agg["p_mw"].values
        Q_D[agg.index.values.astype(int)] = agg["q_mvar"].values

        total_p = P_D.sum()
        xp = (P_D - meta["P_mu"]) / meta["P_sig"]
        xq = (Q_D - meta["Q_mu"]) / meta["Q_sig"]
        p_frac = P_D / (total_p + eps)
        log_scale = np.clip(np.log((total_p + eps) / (meta["nominal_total"] + eps)), -3, 3)

        gen_df = tables["gen"]
        p_max_now = np.array([float(gen_df.at[gi, "max_p_mw"]) for gi in ai])
        p_max_ratio = np.clip(p_max_now / np.maximum(meta["p_max_nom"], eps), 0, 2)

        c1_now = np.zeros(len(ai))
        c2_now = np.zeros(len(ai))
        poly_cost_df = tables["poly_cost"]
        if len(poly_cost_df):
            for k, gi in enumerate(ai):
                mask = (poly_cost_df["element"] == gi) & (poly_cost_df["et"] == "gen")
                if mask.any():
                    row = poly_cost_df[mask].iloc[0]
                    c1_now[k] = float(row.get("cp1_eur_per_mw", 0.0))
                    c2_now[k] = float(row.get("cp2_eur_per_mw2", 0.0))
        c1_ratio = np.clip(c1_now / np.where(np.abs(meta["c1_nom"]) > eps, meta["c1_nom"], eps), 0, 5)
        c2_ratio = np.clip(c2_now / np.where(np.abs(meta["c2_nom"]) > eps, meta["c2_nom"], eps), 0, 5)

        trafo_df = tables["trafo"]
        t_now = np.array([float(trafo_df.at[t, "max_loading_percent"]) for t in meta["trafo_idx"]])
        trafo_ratio = np.clip(t_now / np.maximum(meta["trafo_nom"], eps), 0, 2)

        x = np.concatenate([xp, xq, p_frac, [log_scale], p_max_ratio, c1_ratio, c2_ratio, trafo_ratio])
        x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

        res_gen_df = tables["res_gen"]
        res_bus_df = tables["res_bus"]
        PG = np.array([float(res_gen_df.at[gi, "p_mw"]) for gi in ai])
        sv = float(res_bus_df.at[sl, "vm_pu"])
        vm_pv = res_bus_df["vm_pu"].values[pv].astype(float)

        PG_n = (PG - meta["PG_lo"]) / (meta["PG_hi"] - meta["PG_lo"] + eps)
        sv_n = (sv - meta["Slack_lo"]) / (meta["Slack_hi"] - meta["Slack_lo"] + eps)
        vm_n = (vm_pv - meta["Vm_lo"]) / (meta["Vm_hi"] - meta["Vm_lo"] + eps)
        y = np.concatenate([PG_n, [sv_n], vm_n])
        if not np.isfinite(y).all():
            return None
        return x, y
    except Exception:
        return None

def _build_one_sample_worker(path):
    return _build_one_sample(path, _SAMPLE_META)


def _build_samples(paths, meta, build_workers=0):
    if build_workers > 1:
        chunksize = max(1, len(paths) // (build_workers * 4))
        with _mp_context().Pool(
            processes=build_workers,
            initializer=_worker_init,
            initargs=(meta,),
        ) as pool:
            rows = pool.map(
                _build_one_sample_worker,
                paths,
                chunksize=chunksize,
            )
    else:
        rows = [_build_one_sample(path, meta) for path in paths]

    #sanity check: if any issues with scenarios
    failed = [p for p, row in zip(paths, rows) if row is None]
    if failed:
        raise RuntimeError(
            "{} scenario(s) failed to build (first: {}). Extended data "
            "generation does not drop samples; remove or fix these scenarios."
            .format(len(failed), failed[0])
        )

    Xs, Ys = zip(*rows)
    return (torch.tensor(np.vstack(Xs), dtype=torch.float32),
            torch.tensor(np.vstack(Ys), dtype=torch.float32))


# -----------------------------------------------------------------------
# Fast nominal build: reuse the saved non-extended dataset
# -----------------------------------------------------------------------
#
# The non-extended training dataset already stores the loads (X) and OPF
# labels (Y) for every nominal scenario.  For nominal data the extra context
# features are trivial: the capacity / cost / transformer ratios are all 1
# (nominal scenarios change only loads), and p_frac / log_scale are pure
# functions of the loads.  So the extended nominal dataset can be assembled
# directly from the saved non-extended one with NO network reads and NO
# dropped samples.  The only requirement is that both use the same input
# normalization; _adopt_nonext_input_stats arranges exactly that (the output
# bounds already match because they are base-network properties).

def _nonext_train_dataset_path(case):
    return os.path.join(
        "src", "baselines", "DeepOPF", "datasets", case, "training_dataset.pt"
    )


def _adopt_nonext_input_stats(meta, case):
    """
    Make the saved non-extended dataset directly reusable by adopting its input
    normalization stats (P_mu/P_sig/Q_mu/Q_sig) into `meta`.  nominal_total is
    derived consistently as mean(P_mu), matching _build_meta's definition.
    Returns the loaded non-extended dataset dict, or None if it is absent.
    """
    path = _nonext_train_dataset_path(case)
    if not os.path.exists(path):
        return None
    nd = torch.load(path, map_location="cpu")
    nm = nd["meta"]
    for k in ("P_mu", "P_sig", "Q_mu", "Q_sig"):
        meta[k] = np.asarray(nm[k], dtype=float)
    meta["nominal_total"] = float(np.mean(meta["P_mu"]))
    return nd


def _extend_nominal_from_saved(nd, meta):
    """
    Assemble extended NOMINAL (X, Y) from a saved non-extended dataset dict,
    reusing loads and labels verbatim and appending p_frac, log_scale, and unit
    ratios.  Feature formulas match _build_one_sample exactly (so this path and
    the net-reading path are on the same scale).  No reads, no drops.
    """
    eps = 1e-8
    if "dataset" in nd:
        Xb = nd["dataset"].tensors[0].numpy().astype(np.float64)
        Y = nd["dataset"].tensors[1].numpy().astype(np.float32)
    else:
        Xb = nd["X"].numpy().astype(np.float64)
        Y = nd["Y"].numpy().astype(np.float32)

    n_bus = meta["n_bus"]
    n_gen = len(meta["active_gen_idx"])
    n_tr = len(meta["trafo_idx"])
    P_mu = np.asarray(meta["P_mu"]); P_sig = np.asarray(meta["P_sig"])

    Pz = Xb[:, :n_bus]
    Qz = Xb[:, n_bus:2 * n_bus]
    P_D = P_mu[None, :] + Pz * P_sig[None, :]          # recover raw loads
    total = P_D.sum(axis=1, keepdims=True)
    p_frac = P_D / (total + eps)
    log_scale = np.clip(
        np.log((total[:, 0] + eps) / (meta["nominal_total"] + eps)), -3, 3
    )[:, None]
    ones = lambda c: np.ones((Xb.shape[0], c), dtype=np.float64)  # nominal ratios = 1

    X = np.concatenate(
        [Pz, Qz, p_frac, log_scale, ones(n_gen), ones(n_gen), ones(n_gen), ones(n_tr)],
        axis=1,
    )
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


# -----------------------------------------------------------------------
# Parallel penalty helpers  (no dependency on train_augmented)
# -----------------------------------------------------------------------

_PEN_CTX = None


def _pen_worker_init(ctx):
    global _PEN_CTX
    _PEN_CTX = ctx
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _pen_worker_eval(task):
    y_np, x_np = task
    return _pen_one(y_np, x_np, _PEN_CTX)


def _pen_one(y_np, x_np, C):
    """Numpy version of PenaltyACPF._penalty_single_sample from train_model.py."""
    base_net = C["base_net"]
    n_pg = len(C["active_gen_idx"])
    n_pv = len(C["pv_bus_list"])

    PG     = C["PG_lo"]    + y_np[0:n_pg]            * (C["PG_hi"]    - C["PG_lo"])
    SlackV = C["Slack_lo"] + y_np[n_pg]               * (C["Slack_hi"] - C["Slack_lo"])
    Vm_PV  = C["Vm_lo"]    + y_np[n_pg+1:n_pg+1+n_pv] * (C["Vm_hi"]   - C["Vm_lo"])

    n_bus = len(base_net.bus)
    P_D = C["P_mu"] + x_np[0:n_bus]       * C["P_sig"]
    Q_D = C["Q_mu"] + x_np[n_bus:2*n_bus] * C["Q_sig"]

    net = copy.deepcopy(base_net)
    load_map = C["load_bus_to_idxs"]
    pv_pos   = C["pv_pos"]

    if len(net.load):
        for bus in net.bus.index:
            idxs = load_map.get(int(bus))
            if not idxs:
                continue
            net.load.loc[idxs, "p_mw"]   = float(P_D[int(bus)]) / len(idxs)
            net.load.loc[idxs, "q_mvar"] = float(Q_D[int(bus)]) / len(idxs)

    for k, gi in enumerate(C["active_gen_idx"]):
        net.gen.at[gi, "p_mw"] = float(PG[k])

    if len(net.ext_grid):
        eg = net.ext_grid.index[net.ext_grid["bus"] == C["slack_bus"]]
        if len(eg):
            net.ext_grid.at[eg[0], "vm_pu"] = float(SlackV)

    for k, gi in enumerate(C["active_gen_idx"]):
        pos = pv_pos.get(int(net.gen.at[gi, "bus"]))
        if pos is not None:
            net.gen.at[gi, "vm_pu"] = float(Vm_PV[pos])

    PEN_FAIL = 1.0
    prep_constraints(net)
    try:
        run_pf(net, vm_init=C.get("vm_init"), va_init=C.get("va_init"))
    except Exception:
        return PEN_FAIL
    if not getattr(net, "converged", True):
        return PEN_FAIL
    pen = compute_penalty_from_net(net)
    return float(pen) if np.isfinite(pen) else PEN_FAIL


def _eval_pen_batch(y_np, x_np, C, pool, penalty_workers):
    tasks = list(zip(y_np, x_np))
    if pool is not None and penalty_workers > 1 and len(tasks) > 1:
        chunksize = max(1, len(tasks) // (penalty_workers * 4))
        vals = pool.map(_pen_worker_eval, tasks, chunksize=chunksize)
    else:
        vals = [_pen_one(y, x, C) for y, x in tasks]
    return np.asarray(vals, dtype=np.float32)


class _PenaltyFn(Function):
    """ zero-order penalty, pool-aware
    for faster compute"""

    @staticmethod
    def forward(ctx, y_norm, x_norm, C, pool, penalty_workers):
        ctx.save_for_backward(y_norm, x_norm)
        ctx.C = C
        ctx.pool = pool
        ctx.pw = int(penalty_workers)
        vals = _eval_pen_batch(
            y_norm.detach().cpu().numpy(),
            x_norm.detach().cpu().numpy(),
            C, pool, int(penalty_workers),
        )
        return torch.as_tensor(vals.sum(), dtype=torch.float32, device=y_norm.device)

    @staticmethod
    def backward(ctx, grad_output):
        y_norm, x_norm = ctx.saved_tensors
        C, pool, pw = ctx.C, ctx.pool, ctx.pw
        device = y_norm.device
        B, _ = y_norm.shape

        v = torch.randn_like(y_norm)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        h = 1e-4

        y_plus  = (y_norm + h * v).detach().cpu().numpy()
        y_minus = (y_norm - h * v).detach().cpu().numpy()
        x_np    = x_norm.detach().cpu().numpy()

        p_plus  = _eval_pen_batch(y_plus,  x_np, C, pool, pw)
        p_minus = _eval_pen_batch(y_minus, x_np, C, pool, pw)

        dir_deriv = torch.as_tensor((p_plus - p_minus) / (2.0 * h),
                                    dtype=torch.float32, device=device)
        S  = len(C["active_gen_idx"]) + len(C["pv_bus_list"]) + 1
        go = float(grad_output.item())
        grad_y = go * (S * dir_deriv).view(B, 1) * v
        return grad_y, torch.zeros_like(x_norm), None, None, None


def _train_model_with_workers(
    base_net, X_t, Y_t, meta,
    epochs=20, batch_size=3072, lr=1e-3,
    device="cuda", w1=1.0, w2=0.1, penalty_workers=0,
):
    """Identical to train_model.train_penalty_model but penalty PFs run in parallel."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    _, in_dim = X_t.shape
    out_dim   = len(meta["active_gen_idx"]) + 1 + len(meta["pv_bus_list"])

    model = DeepOPFPenaltyNet(in_dim, hidden=vars.model_hidden, out_dim=out_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss(reduction="mean")

    load_bus_to_idxs = {}
    if len(base_net.load):
        buses = base_net.load["bus"].to_numpy()
        idxs  = base_net.load.index.to_numpy()
        for b in base_net.bus.index.to_numpy():
            m = (buses == b)
            if m.any():
                load_bus_to_idxs[int(b)] = idxs[m]

    pv_pos = {int(b): i for i, b in enumerate(meta["pv_bus_list"])}
    vm_init, va_init = build_hist_initialization(base_net, meta, Y_t)

    C = {
        "base_net":         base_net,
        "P_mu":  meta["P_mu"],      "P_sig":  meta["P_sig"],
        "Q_mu":  meta["Q_mu"],      "Q_sig":  meta["Q_sig"],
        "active_gen_idx":   meta["active_gen_idx"],
        "pv_bus_list":      meta["pv_bus_list"],
        "slack_bus":        meta["slack_bus"],
        "PG_lo": meta["PG_lo"],     "PG_hi":    meta["PG_hi"],
        "Slack_lo": meta["Slack_lo"], "Slack_hi": meta["Slack_hi"],
        "Vm_lo": meta["Vm_lo"],     "Vm_hi":    meta["Vm_hi"],
        "load_bus_to_idxs": load_bus_to_idxs,
        "pv_pos":    pv_pos,
        "vm_init":   vm_init,
        "va_init":   va_init,
    }

    pool = None
    if penalty_workers > 1:
        pool = _mp_context().Pool(
            processes=penalty_workers,
            initializer=_pen_worker_init,
            initargs=(C,),
        )

    dl = DataLoader(TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True)

    try:
        for ep in range(epochs):
            model.train()
            total_mse = total_pen = 0.0
            step = 0
            for xb, yb in dl:
                step += 1
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                y_pred = model(xb)
                pen  = _PenaltyFn.apply(y_pred, xb, C, pool, penalty_workers)
                mse  = criterion(y_pred, yb)
                loss = w1 * mse + w2 * pen
                if step % 50 == 0:
                    print(f"[ep {ep+1:03d} step {step:04d}] "
                          f"mse={mse.item():.6e}  "
                          f"pen_sum={pen.item():.6e}  "
                          f"pen_per_sample={pen.item()/xb.size(0):.6e}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_mse += float(mse.detach().cpu())
                total_pen += float(pen.detach().cpu())
            print(f"[epoch {ep+1:03d}] mse = {total_mse/len(dl):.6f} ; "
                  f"penalty = {total_pen/len(dl):.6f}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--groups", default="in_distribution")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--gen_data", action="store_true", default=False,
                        help="Build dataset from scenario files and save. "
                             "If False (default), load the previously saved dataset.")
    parser.add_argument("--build_workers", type=int, default=-1,
                        help="-1=auto, 0/1=serial, >1=parallel data-loading workers")
    parser.add_argument("--penalty_workers", type=int, default=-1,
                        help="-1=auto, 0/1=serial, >1=parallel PF penalty workers during training")
    args = parser.parse_args()

    ncpu = os.cpu_count() or 1
    if args.build_workers == -1:
        args.build_workers = max(1, min(8, ncpu - 1))
    if args.penalty_workers == -1:
        args.penalty_workers = max(1, min(8, ncpu - 1))

    groups = _parse_groups(args.groups)
    tag = _groups_tag(groups)
    dataset_dir  = os.path.join("src", "baselines", "DeepOPF", "datasets", args.case)
    dataset_path = os.path.join(dataset_dir, "ext_input_{}_dataset.pt".format(tag))

    base_net, _ = util.get_base_net(args.case, "in_distribution")

    if args.gen_data:
        print("Build workers: {}  Penalty workers: {}".format(args.build_workers, args.penalty_workers))
        print("Building meta from {} nominal scenarios with {} worker(s)...".format(
            args.case, args.build_workers))
        meta = _build_meta(args.case, args.build_workers)
        nd = _adopt_nonext_input_stats(meta, args.case)
        if nd is not None:
            print("Adopted non-extended input stats; nominal group will reuse "
                  "the saved non-extended dataset (no net reads).")

        all_X, all_Y = [], []
        for g in groups:
            if g == "in_distribution" and nd is not None:
                X, Y = _extend_nominal_from_saved(nd, meta)
                print("  in_distribution (reused non-extended dataset): {} samples".format(len(X)))
            else:
                X, Y = _build_samples(_load_paths(args.case, g), meta, args.build_workers)
                print("  {}: {} samples".format(g, len(X)))
            all_X.append(X)
            all_Y.append(Y)
        X = torch.cat(all_X)
        Y = torch.cat(all_Y)
        print("  Total: {} samples  in_dim={}  out_dim={}".format(len(X), X.shape[1], Y.shape[1]))

        os.makedirs(dataset_dir, exist_ok=True)
        torch.save({
            "dataset": TensorDataset(X, Y),
            "X": X, "Y": Y, "meta": meta, "groups": groups,
            "in_dim": X.shape[1], "out_dim": Y.shape[1],
        }, dataset_path)
        print("Saved dataset -> {}".format(dataset_path))
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                "Dataset not found: {}\n"
                "Run with --gen_data first to build it.".format(dataset_path)
            )
        loaded = torch.load(dataset_path, map_location="cpu")
        X    = loaded["X"].float()
        Y    = loaded["Y"].float()
        meta = loaded["meta"]
        print("Loaded dataset from {} ({} samples)".format(dataset_path, len(X)))

    model_dir = os.path.join("src", "baselines", "DeepOPF", "model", args.case)
    os.makedirs(model_dir, exist_ok=True)
    ckpt = os.path.join(model_dir, "trained_model_ext_input.pth")

    model = _train_model_with_workers(
        base_net, X, Y, meta,
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, device=args.device,
        w1=args.w1, w2=args.w2,
        penalty_workers=args.penalty_workers,
    )

    torch.save({"state_dict": model.state_dict(), "meta": meta,
                "in_dim": X.shape[1], "groups": groups,
                "model_class": "DeepOPFPenaltyNet"}, ckpt)
    print("Saved model -> {}".format(ckpt))


if __name__ == "__main__":
    mp.freeze_support()
    main()