import os
import copy
import atexit
import inspect
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader

import pandapower as pp

from src import util
from src import global_vars as gvars
from src.baselines.DeepOPF import vars
from src.baselines.DeepOPF.make_dataset import aggregate_bus_vector


# ============================================================
# Runtime / threading
# ============================================================

def set_runtime_threads(cpu_threads: int = 0, interop_threads: int = 0):
    """
    Must be called before heavy eager/autograd work starts.
    """
    if cpu_threads and cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
        torch.set_num_threads(int(cpu_threads))

    if interop_threads and interop_threads > 0:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            # can only be called once per process before inter-op work begins
            pass


def auto_main_cpu_threads():
    n = os.cpu_count() or 1
    return max(1, min(4, n))


def auto_penalty_workers():
    n = os.cpu_count() or 1
    return max(1, min(8, n - 1))


# ============================================================
# Dataset helpers
# ============================================================

def load_saved_dataset(fname: str):
    loaded = torch.load(fname + ".pt", map_location="cpu")
    dataset = loaded["dataset"]
    meta = loaded["meta"]

    if hasattr(dataset, "tensors") and len(dataset.tensors) == 2:
        X_t, Y_t = dataset.tensors
    else:
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        X_list, Y_list = [], []
        for xb, yb in loader:
            X_list.append(xb)
            Y_list.append(yb)
        X_t = torch.cat(X_list, dim=0)
        Y_t = torch.cat(Y_list, dim=0)

    return X_t.float(), Y_t.float(), dataset, meta

def build_stressed_training_data_loader(
    base_net,
    case,
    fname,
    batch_size=128,
    scen_subdir="training_stressed",
):
    train_meta_path = f"src/baselines/DeepOPF/datasets/{case}/training_dataset.pt"
    loaded = torch.load(train_meta_path, map_location="cpu")
    train_meta = loaded["meta"]

    P_mu = np.asarray(train_meta["P_mu"], dtype=float)
    P_sig = np.asarray(train_meta["P_sig"], dtype=float)
    Q_mu = np.asarray(train_meta["Q_mu"], dtype=float)
    Q_sig = np.asarray(train_meta["Q_sig"], dtype=float)

    PG_lo_train = np.asarray(train_meta["PG_lo"], dtype=float)
    PG_hi_train = np.asarray(train_meta["PG_hi"], dtype=float)
    Vm_lo_train = np.asarray(train_meta["Vm_lo"], dtype=float)
    Vm_hi_train = np.asarray(train_meta["Vm_hi"], dtype=float)

    slack_bus = int(train_meta["slack_bus"])
    Slack_lo_train = float(train_meta["Slack_lo"])
    Slack_hi_train = float(train_meta["Slack_hi"])

    active_gen_idx = list(train_meta["active_gen_idx"])
    pv_bus_list = list(train_meta["pv_bus_list"])

    scen_root = os.path.join("src", "scenarios", case, scen_subdir)
    if not os.path.isdir(scen_root):
        raise FileNotFoundError(f"Stressed scenario root not found: {scen_root}")

    stress_case_dirs = [
        os.path.join(scen_root, d)
        for d in sorted(os.listdir(scen_root))
        if os.path.isdir(os.path.join(scen_root, d))
    ]
    if len(stress_case_dirs) == 0:
        raise RuntimeError(f"No stress-case folders found under: {scen_root}")

    all_paths = []
    all_groups = []

    for group_dir in stress_case_dirs:
        group_name = os.path.basename(group_dir)
        manifest_path = os.path.join(group_dir, "manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        if "status" not in manifest.columns or "file" not in manifest.columns:
            raise ValueError(f"Manifest must contain columns ['status', 'file']: {manifest_path}")

        manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)

        for rel_file in manifest["file"].tolist():
            path = os.path.join(group_dir, str(rel_file))
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scenario file listed in manifest not found: {path}")
            all_paths.append(path)
            all_groups.append(group_name)

    if len(all_paths) == 0:
        raise RuntimeError(f"No valid stressed scenario files found under: {scen_root}")

    P_D_list, Q_D_list = [], []
    PG_active_list, Vm_list, SlackV_list = [], [], []

    for path in all_paths:
        net = util.load_net(path)

        P_D_vec = aggregate_bus_vector(net, "p_mw")
        Q_D_vec = aggregate_bus_vector(net, "q_mvar")

        PG_vec = np.array(
            [float(net.res_gen.at[gi, "p_mw"]) for gi in active_gen_idx],
            dtype=float
        )
        slack_v = float(net.res_bus.at[slack_bus, "vm_pu"])
        Vm = net.res_bus.vm_pu.values.astype(float)
        Vm_pv = Vm[pv_bus_list]

        P_D_list.append(P_D_vec)
        Q_D_list.append(Q_D_vec)
        PG_active_list.append(PG_vec)
        SlackV_list.append(slack_v)
        Vm_list.append(Vm_pv)

    num_total_bus = len(base_net.bus)
    P_D = np.vstack(P_D_list) if P_D_list else np.empty((0, num_total_bus))
    Q_D = np.vstack(Q_D_list) if Q_D_list else np.empty((0, num_total_bus))
    PG_active = np.vstack(PG_active_list) if PG_active_list else np.empty((0, len(active_gen_idx)))
    SlackV = np.asarray(SlackV_list, dtype=float).reshape(-1, 1)
    Vm_pv = np.vstack(Vm_list) if Vm_list else np.empty((0, len(pv_bus_list)))

    X = np.concatenate([
        (P_D - P_mu) / (P_sig + 1e-8),
        (Q_D - Q_mu) / (Q_sig + 1e-8),
    ], axis=1)

    PG_norm = (PG_active - PG_lo_train[None, :]) / (PG_hi_train[None, :] - PG_lo_train[None, :] + 1e-8)
    Slack_norm = (SlackV - Slack_lo_train) / (Slack_hi_train - Slack_lo_train + 1e-8)
    Vm_norm = (Vm_pv - Vm_lo_train[None, :]) / (Vm_hi_train[None, :] - Vm_lo_train[None, :] + 1e-8)
    Y = np.concatenate([PG_norm, Slack_norm, Vm_norm], axis=1)

    # global shuffle across ALL stress folders
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    shuffled_groups = [all_groups[i] for i in perm.tolist()]

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    group_counts = {}
    for g in shuffled_groups:
        group_counts[g] = group_counts.get(g, 0) + 1

    artifacts = {
        "which_data": scen_subdir,
        "stress_groups": sorted(group_counts.keys()),
        "stress_group_counts": group_counts,
        "P_mu": P_mu,
        "P_sig": P_sig,
        "Q_mu": Q_mu,
        "Q_sig": Q_sig,
        "active_gen_idx": active_gen_idx,
        "pv_bus_list": pv_bus_list,
        "PG_lo": PG_lo_train,
        "PG_hi": PG_hi_train,
        "slack_bus": slack_bus,
        "Slack_lo": Slack_lo_train,
        "Slack_hi": Slack_hi_train,
        "Vm_lo": Vm_lo_train,
        "Vm_hi": Vm_hi_train,
    }

    torch.save({"dataset": dataset, "meta": artifacts}, fname + ".pt")
    print(f"[stressed-data] loaded {X.shape[0]} samples from {len(group_counts)} stress folders")
    print(f"[stressed-data] per-folder counts: {group_counts}")

    return loader, artifacts

# ============================================================
# Power flow / penalty helpers
# ============================================================

def prep_constraints(net):
    if "min_vm_pu" not in net.bus:
        net.bus["min_vm_pu"] = 0.95
    if "max_vm_pu" not in net.bus:
        net.bus["max_vm_pu"] = 1.05

    for c in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if c not in net.gen:
            net.gen[c] = np.nan

    net.gen["min_p_mw"] = net.gen["min_p_mw"].fillna(0.0)
    net.gen["max_p_mw"] = net.gen["max_p_mw"].fillna(np.maximum(1.0, net.gen["p_mw"] * 1.2))
    net.gen["min_q_mvar"] = net.gen["min_q_mvar"].fillna(-1e3)
    net.gen["max_q_mvar"] = net.gen["max_q_mvar"].fillna(+1e3)

    if len(net.line):
        if "max_loading_percent" not in net.line:
            net.line["max_loading_percent"] = 100.0
        net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(100.0)

    if len(net.trafo):
        if "max_loading_percent" not in net.trafo:
            net.trafo["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = net.trafo["max_loading_percent"].fillna(100.0)

    if hasattr(net, "trafo3w") and len(net.trafo3w):
        if "max_loading_percent" not in net.trafo3w:
            net.trafo3w["max_loading_percent"] = 100.0
        net.trafo3w["max_loading_percent"] = net.trafo3w["max_loading_percent"].fillna(100.0)


def build_hist_initialization(base_net, meta, Y_t):
    n_bus = len(base_net.bus)
    bus_index = base_net.bus.index.to_numpy()
    bus_to_pos = {int(b): i for i, b in enumerate(bus_index)}

    pv_bus_list = meta["pv_bus_list"]
    slack_bus = int(meta["slack_bus"])

    n_pg = len(meta["active_gen_idx"])
    n_pv = len(pv_bus_list)

    slack_norm = Y_t[:, n_pg]
    vm_pv_norm = Y_t[:, n_pg + 1: n_pg + 1 + n_pv]

    Slack_lo, Slack_hi = meta["Slack_lo"], meta["Slack_hi"]
    Vm_lo, Vm_hi = meta["Vm_lo"], meta["Vm_hi"]

    SlackV_mean = float(Slack_lo + slack_norm.mean().item() * (Slack_hi - Slack_lo))
    Vm_PV_mean = (Vm_lo + vm_pv_norm.mean(dim=0).cpu().numpy() * (Vm_hi - Vm_lo)).ravel()

    vm_init = np.ones(n_bus, dtype=float)
    vm_init[bus_to_pos[slack_bus]] = SlackV_mean
    for i, b in enumerate(pv_bus_list):
        vm_init[bus_to_pos[int(b)]] = float(Vm_PV_mean[i])

    va_init = np.zeros(n_bus, dtype=float)
    return vm_init, va_init


def resolve_runpp_kwargs(enable_numba=True, enable_lightsim2grid=True):
    sig = inspect.signature(pp.runpp)
    kwargs = {
        "algorithm": "nr",
        "enforce_q_lims": False,
        "calculate_voltage_angles": True,
    }

    if "numba" in sig.parameters:
        kwargs["numba"] = bool(enable_numba)

    if enable_lightsim2grid and "lightsim2grid" in sig.parameters:
        kwargs["lightsim2grid"] = "auto"

    return kwargs


def run_pf(net, vm_init=None, va_init=None, runpp_kwargs=None):
    kwargs = dict(runpp_kwargs or {})
    if vm_init is not None or va_init is not None:
        pp.runpp(
            net,
            init="auto",
            init_vm_pu=vm_init,
            init_va_degree=va_init,
            **kwargs,
        )
    else:
        pp.runpp(net, **kwargs)


def compute_penalty_from_net(net, ctol=gvars.ctol) -> float:
    pen = 0.0
    baseMVA = float(net.sn_mva)

    if len(net.gen):
        g = net.gen
        rg = net.res_gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool)

        if {"min_p_mw", "max_p_mw"}.issubset(g.columns):
            p = rg["p_mw"].to_numpy()[on]
            pmin = g["min_p_mw"].to_numpy()[on]
            pmax = g["max_p_mw"].to_numpy()[on]
            u = np.maximum(p - pmax, 0.0) / baseMVA
            l = np.maximum(pmin - p, 0.0) / baseMVA
            u[u < ctol] = 0.0
            l[l < ctol] = 0.0
            pen += u.sum() + l.sum()

        if {"min_q_mvar", "max_q_mvar"}.issubset(g.columns):
            q = rg["q_mvar"].to_numpy()[on]
            qmin = g["min_q_mvar"].to_numpy()[on]
            qmax = g["max_q_mvar"].to_numpy()[on]
            u = np.maximum(q - qmax, 0.0) / baseMVA
            l = np.maximum(qmin - q, 0.0) / baseMVA
            u[u < ctol] = 0.0
            l[l < ctol] = 0.0
            pen += u.sum() + l.sum()

    if len(net.bus):
        v = net.res_bus["vm_pu"].to_numpy()

        v_lo = net.bus.get("min_vm_pu", 0.95)
        v_hi = net.bus.get("max_vm_pu", 1.05)
        v_lo = v_lo if np.isscalar(v_lo) else v_lo.to_numpy()
        v_hi = v_hi if np.isscalar(v_hi) else v_hi.to_numpy()

        u = np.maximum(v - v_hi, 0.0)
        l = np.maximum(v_lo - v, 0.0)
        u[u < ctol] = 0.0
        l[l < ctol] = 0.0
        pen += u.sum() + l.sum()

    if len(net.line):
        rl = net.res_line
        S_from = np.sqrt(rl["p_from_mw"] ** 2 + rl["q_from_mvar"] ** 2).to_numpy()
        S_to = np.sqrt(rl["p_to_mw"] ** 2 + rl["q_to_mvar"] ** 2).to_numpy()
        lpct = rl["loading_percent"].to_numpy()

        S_from = np.nan_to_num(S_from, nan=0.0, posinf=0.0, neginf=0.0)
        S_to = np.nan_to_num(S_to, nan=0.0, posinf=0.0, neginf=0.0)
        lpct = np.nan_to_num(lpct, nan=0.0, posinf=0.0, neginf=0.0)
        lpct = np.maximum(lpct, 1e-3)

        eps = 1e-12
        Smax_from = np.where(lpct > eps, S_from * 100.0 / lpct, np.inf)
        Smax_to = np.where(lpct > eps, S_to * 100.0 / lpct, np.inf)

        over_from = np.maximum(S_from - Smax_from, 0.0) / baseMVA
        over_to = np.maximum(S_to - Smax_to, 0.0) / baseMVA
        over_from[over_from < ctol] = 0.0
        over_to[over_to < ctol] = 0.0
        pen += over_from.sum() + over_to.sum()

    if len(net.trafo):
        rt = net.res_trafo
        S_hv = np.sqrt(rt["p_hv_mw"] ** 2 + rt["q_hv_mvar"] ** 2).to_numpy()
        lpct = rt["loading_percent"].to_numpy()
        eps = 1e-12
        Smax = np.where(lpct > eps, S_hv * 100.0 / lpct, np.inf)
        over = np.maximum(S_hv - Smax, 0.0) / baseMVA
        over[over < ctol] = 0.0
        pen += over.sum()

    if hasattr(net, "trafo3w") and len(net.trafo3w):
        rt3 = net.res_trafo3w
        S_hv = np.sqrt(rt3["p_h_mw"] ** 2 + rt3["q_h_mvar"] ** 2).to_numpy()
        lpct = rt3["loading_percent"].to_numpy()
        eps = 1e-12
        Smax = np.where(lpct > eps, S_hv * 100.0 / lpct, np.inf)
        over = np.maximum(S_hv - Smax, 0.0) / baseMVA
        over[over < ctol] = 0.0
        pen += over.sum()

    return float(pen)


# ============================================================
# Penalty worker pool
# ============================================================

_PENALTY_WORKER_CTX = None


def _init_penalty_worker(ctx_dict):
    global _PENALTY_WORKER_CTX
    _PENALTY_WORKER_CTX = ctx_dict

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def _penalty_single_sample_np(y_norm_np, x_norm_np, C):
    base_net = C["base_net"]
    bus_to_pos = C["bus_to_pos"]

    n_pg = len(C["active_gen_idx"])
    n_pv = len(C["pv_bus_list"])

    PG_norm = y_norm_np[0:n_pg]
    Slack_norm = y_norm_np[n_pg]
    Vm_norm = y_norm_np[n_pg + 1: n_pg + 1 + n_pv]

    PG = C["PG_lo"] + PG_norm * (C["PG_hi"] - C["PG_lo"])
    SlackV = C["Slack_lo"] + Slack_norm * (C["Slack_hi"] - C["Slack_lo"])
    Vm_PV = C["Vm_lo"] + Vm_norm * (C["Vm_hi"] - C["Vm_lo"])

    n_bus = len(C["bus_index"])
    Pz = x_norm_np[0:n_bus]
    Qz = x_norm_np[n_bus: 2 * n_bus]
    P_D = C["P_mu"] + Pz * C["P_sig"]
    Q_D = C["Q_mu"] + Qz * C["Q_sig"]

    net = copy.deepcopy(base_net)

    if len(net.load):
        for bus in C["buses_with_load"]:
            idxs = C["load_bus_to_idxs"][int(bus)]
            pos = bus_to_pos[int(bus)]
            share_p = float(P_D[pos]) / len(idxs)
            share_q = float(Q_D[pos]) / len(idxs)
            net.load.loc[idxs, "p_mw"] = share_p
            net.load.loc[idxs, "q_mvar"] = share_q

    for k, gi in enumerate(C["active_gen_idx"]):
        net.gen.at[gi, "p_mw"] = float(PG[k])

    if C["slack_ext_grid_idx"] is not None:
        net.ext_grid.at[C["slack_ext_grid_idx"], "vm_pu"] = float(SlackV)

    for k, gi in enumerate(C["active_gen_idx"]):
        bus_k = int(C["active_gen_buses"][k])
        pos = C["pv_pos"].get(bus_k)
        if pos is not None:
            net.gen.at[gi, "vm_pu"] = float(Vm_PV[pos])

    PEN_FAIL = 1.0
    try:
        run_pf(
            net,
            vm_init=C.get("vm_init"),
            va_init=C.get("va_init"),
            runpp_kwargs=C["runpp_kwargs"],
        )
    except Exception:
        return PEN_FAIL

    if not getattr(net, "converged", True):
        return PEN_FAIL

    pen = compute_penalty_from_net(net)
    if not np.isfinite(pen):
        pen = PEN_FAIL

    return float(pen)

def _penalty_worker_eval(task):
    y_np, x_np = task
    return _penalty_single_sample_np(y_np, x_np, _PENALTY_WORKER_CTX)

def create_penalty_pool(ctx_dict, penalty_workers: int, use_cuda: bool):
    if penalty_workers <= 1:
        return None

    if use_cuda:
        ctx = mp.get_context("spawn")
    else:
        start_methods = mp.get_all_start_methods()
        start_method = "fork" if "fork" in start_methods else "spawn"
        ctx = mp.get_context(start_method)

    pool = ctx.Pool(
        processes=int(penalty_workers),
        initializer=_init_penalty_worker,
        initargs=(ctx_dict,),
    )
    atexit.register(close_penalty_pool, pool)
    return pool

def close_penalty_pool(pool):
    if pool is None:
        return
    try:
        pool.close()
    except Exception:
        pass
    try:
        pool.join()
    except Exception:
        pass


def eval_penalty_batch(y_norm, x_norm, ctx_dict, penalty_pool=None, penalty_workers=0):
    device = y_norm.device
    y_np = y_norm.detach().cpu().numpy()
    x_np = x_norm.detach().cpu().numpy()

    tasks = list(zip(y_np, x_np))

    if penalty_pool is not None and penalty_workers > 1 and len(tasks) > 1:
        chunksize = max(1, len(tasks) // (penalty_workers * 4))
        vals = penalty_pool.map(_penalty_worker_eval, tasks, chunksize=chunksize)
    else:
        vals = [_penalty_single_sample_np(y_i, x_i, ctx_dict) for y_i, x_i in tasks]

    vals = np.asarray(vals, dtype=np.float32)
    return torch.as_tensor(vals, dtype=torch.float32, device=device)


# ============================================================
# Custom autograd penalty
# ============================================================

class PenaltyACPF(Function):
    @staticmethod
    def forward(ctx, y_norm, x_norm, ctx_dict, penalty_pool, penalty_workers):
        ctx.save_for_backward(y_norm, x_norm)
        ctx.ctx_dict = ctx_dict
        ctx.penalty_pool = penalty_pool
        ctx.penalty_workers = int(penalty_workers)

        penalties = eval_penalty_batch(
            y_norm=y_norm,
            x_norm=x_norm,
            ctx_dict=ctx_dict,
            penalty_pool=penalty_pool,
            penalty_workers=int(penalty_workers),
        )
        return penalties.sum()

    @staticmethod
    def backward(ctx, grad_output):
        y_norm, x_norm = ctx.saved_tensors
        C = ctx.ctx_dict
        device = y_norm.device
        B, _ = y_norm.shape

        v = torch.randn_like(y_norm)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        h = 1e-4

        y_plus = y_norm + h * v
        y_minus = y_norm - h * v

        p_plus = eval_penalty_batch(
            y_norm=y_plus,
            x_norm=x_norm,
            ctx_dict=C,
            penalty_pool=ctx.penalty_pool,
            penalty_workers=ctx.penalty_workers,
        )
        p_minus = eval_penalty_batch(
            y_norm=y_minus,
            x_norm=x_norm,
            ctx_dict=C,
            penalty_pool=ctx.penalty_pool,
            penalty_workers=ctx.penalty_workers,
        )

        dir_deriv = (p_plus - p_minus) / (2.0 * h)

        S = len(C["active_gen_idx"]) + len(C["pv_bus_list"]) + 1
        go = float(grad_output.item())
        grad_y = go * (S * dir_deriv).view(B, 1) * v

        grad_x = torch.zeros_like(x_norm)
        return grad_y, grad_x, None, None, None


# ============================================================
# Model / curriculum helpers
# ============================================================

class DeepOPFPenaltyNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 2 * hidden)
        self.l3 = nn.Linear(2 * hidden, hidden)
        self.l4 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l3(h))
        y = torch.sigmoid(self.l4(h))
        return y


def build_ctx_dict(base_net, meta, Y_hist, runpp_kwargs):
    prepared_base_net = copy.deepcopy(base_net)
    prep_constraints(prepared_base_net)

    bus_index = prepared_base_net.bus.index.to_numpy()
    bus_to_pos = {int(b): i for i, b in enumerate(bus_index)}

    load_bus_to_idxs = {}
    if len(prepared_base_net.load):
        buses = prepared_base_net.load["bus"].to_numpy()
        idxs = prepared_base_net.load.index.to_numpy()
        for b in prepared_base_net.bus.index.to_numpy():
            m = buses == b
            if m.any():
                load_bus_to_idxs[int(b)] = idxs[m]

    buses_with_load = np.array(sorted(load_bus_to_idxs.keys()), dtype=int)
    pv_pos = {int(b): i for i, b in enumerate(meta["pv_bus_list"])}

    slack_ext_grid_idx = None
    if len(prepared_base_net.ext_grid):
        eg_idx = prepared_base_net.ext_grid.index[
            prepared_base_net.ext_grid["bus"] == int(meta["slack_bus"])
        ]
        if len(eg_idx):
            slack_ext_grid_idx = int(eg_idx[0])

    active_gen_buses = np.array(
        [int(prepared_base_net.gen.at[gi, "bus"]) for gi in meta["active_gen_idx"]],
        dtype=int,
    )

    vm_init, va_init = build_hist_initialization(prepared_base_net, meta, Y_hist)

    ctx_dict = {
        "base_net": prepared_base_net,
        "bus_index": bus_index,
        "bus_to_pos": bus_to_pos,
        "P_mu": np.asarray(meta["P_mu"], dtype=float),
        "P_sig": np.asarray(meta["P_sig"], dtype=float),
        "Q_mu": np.asarray(meta["Q_mu"], dtype=float),
        "Q_sig": np.asarray(meta["Q_sig"], dtype=float),
        "active_gen_idx": list(meta["active_gen_idx"]),
        "active_gen_buses": active_gen_buses,
        "pv_bus_list": list(meta["pv_bus_list"]),
        "slack_bus": int(meta["slack_bus"]),
        "slack_ext_grid_idx": slack_ext_grid_idx,
        "PG_lo": np.asarray(meta["PG_lo"], dtype=float),
        "PG_hi": np.asarray(meta["PG_hi"], dtype=float),
        "Slack_lo": float(meta["Slack_lo"]),
        "Slack_hi": float(meta["Slack_hi"]),
        "Vm_lo": np.asarray(meta["Vm_lo"], dtype=float),
        "Vm_hi": np.asarray(meta["Vm_hi"], dtype=float),
        "load_bus_to_idxs": load_bus_to_idxs,
        "buses_with_load": buses_with_load,
        "pv_pos": pv_pos,
        "vm_init": vm_init,
        "va_init": va_init,
        "runpp_kwargs": dict(runpp_kwargs),
    }
    return ctx_dict

def get_epoch_stress_count(epoch, warmup_epochs, ramp_epochs, n_stress_total):
    if n_stress_total <= 0:
        return 0
    if epoch < warmup_epochs:
        return 0
    if ramp_epochs <= 0:
        return int(n_stress_total)
    prog = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return int(round(prog * n_stress_total))


def sample_epoch_dataset_full_union(X_nom, Y_nom, X_str, Y_str, n_stress_use):
    n_nominal = int(X_nom.shape[0])
    n_stress_total = int(X_str.shape[0])

    if n_stress_total <= 0 or n_stress_use <= 0:
        perm = torch.randperm(X_nom.shape[0])
        return X_nom[perm], Y_nom[perm], n_nominal, 0, n_stress_total

    n_stress_use = min(int(n_stress_use), n_stress_total)
    idx_str = torch.randperm(n_stress_total)[:n_stress_use]

    X_epoch = torch.cat([X_nom, X_str[idx_str]], dim=0)
    Y_epoch = torch.cat([Y_nom, Y_str[idx_str]], dim=0)

    perm = torch.randperm(X_epoch.shape[0])
    return X_epoch[perm], Y_epoch[perm], n_nominal, n_stress_use, n_stress_total


def _extract_state_dict_for_loading(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    cleaned = {}
    for k, v in obj.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def initialize_model_weights(model, init_model_path, strict_load=False):
    if init_model_path is None or str(init_model_path).strip() == "":
        print("[init] No init_model_path provided. Training from scratch.")
        return model

    if not os.path.exists(init_model_path):
        raise FileNotFoundError(f"Initial model weights not found: {init_model_path}")

    raw_ckpt = torch.load(init_model_path, map_location="cpu")
    state_dict = _extract_state_dict_for_loading(raw_ckpt)
    incompat = model.load_state_dict(state_dict, strict=strict_load)

    print(f"[init] Loaded initialization weights from: {init_model_path}")
    if not strict_load:
        if len(incompat.missing_keys) > 0:
            print(f"[init] Missing keys: {incompat.missing_keys}")
        if len(incompat.unexpected_keys) > 0:
            print(f"[init] Unexpected keys: {incompat.unexpected_keys}")

    return model


def train_penalty_model_curriculum(
    base_net,
    X_nom,
    Y_nom,
    X_str,
    Y_str,
    meta,
    epochs=5,
    batch_size=3072,
    lr=1e-3,
    device="cuda",
    w1=1.0,
    w2=0.1,
    warmup_epochs=0,
    ramp_epochs=5,
    init_model_path=None,
    strict_load=False,
    cpu_threads=0,
    interop_threads=0,
    data_num_workers=0,
    penalty_workers=0,
    enable_numba=True,
    enable_lightsim2grid=True,
):
    set_runtime_threads(cpu_threads=cpu_threads, interop_threads=interop_threads)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    in_dim = X_nom.shape[1]
    n_pg = len(meta["active_gen_idx"])
    n_pv = len(meta["pv_bus_list"])
    out_dim = n_pg + 1 + n_pv

    model = DeepOPFPenaltyNet(in_dim, hidden=vars.model_hidden, out_dim=out_dim).to(device)
    model = initialize_model_weights(model, init_model_path, strict_load=strict_load)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss(reduction="mean")

    runpp_kwargs = resolve_runpp_kwargs(
        enable_numba=enable_numba,
        enable_lightsim2grid=enable_lightsim2grid,
    )

    Y_hist = torch.cat([Y_nom, Y_str], dim=0) if X_str.shape[0] > 0 else Y_nom
    ctx_dict = build_ctx_dict(base_net, meta, Y_hist, runpp_kwargs=runpp_kwargs)

    penalty_pool = create_penalty_pool(
        ctx_dict,
        penalty_workers=penalty_workers,
        use_cuda=(device.type == "cuda"),
    )

    try:
        for ep in range(epochs):
            n_stress_use = get_epoch_stress_count(ep, warmup_epochs, ramp_epochs, X_str.shape[0])
            X_ep, Y_ep, n_normal, n_stress, n_stress_total = sample_epoch_dataset_full_union(
                X_nom, Y_nom, X_str, Y_str, n_stress_use
            )
            stress_frac = float(n_stress) / float(n_stress_total) if n_stress_total > 0 else 0.0

            dl_kwargs = {
                "dataset": TensorDataset(X_ep, Y_ep),
                "batch_size": batch_size,
                "shuffle": True,
                "num_workers": int(data_num_workers),
                "pin_memory": (device.type == "cuda"),
            }
            if data_num_workers > 0:
                dl_kwargs["persistent_workers"] = True

            dl = DataLoader(**dl_kwargs)

            model.train()
            total_mse = 0.0
            total_pen = 0.0
            step = 0

            for xb, yb in dl:
                step += 1

                xb = xb.to(device, non_blocking=(device.type == "cuda")).float()
                yb = yb.to(device, non_blocking=(device.type == "cuda")).float()

                y_pred = model(xb)
                pen = PenaltyACPF.apply(y_pred, xb, ctx_dict, penalty_pool, int(penalty_workers))
                mse = criterion(y_pred, yb)
                loss = w1 * mse + w2 * pen

                if step % 50 == 0:
                    print(
                        f"[ep {ep+1:03d} step {step:04d}] "
                        f"stress_frac={stress_frac:.4f} "
                        f"mse={mse.item():.6e} "
                        f"pen_sum={pen.item():.6e} "
                        f"pen_per_sample={pen.item()/xb.size(0):.6e}"
                    )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_mse += float(mse.detach().cpu())
                total_pen += float(pen.detach().cpu())

            print(
                f"[epoch {ep+1:03d}] "
                f"stress_frac={stress_frac:.4f} "
                f"normal={n_normal} stressed={n_stress} "
                f"mse={total_mse/len(dl):.6f} "
                f"penalty={total_pen/len(dl):.6f}"
            )

    finally:
        close_penalty_pool(penalty_pool)

    return model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(vars.training_seed)
    np.random.seed(vars.training_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--gen_data", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=0.1)

    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--ramp_epochs", type=int, default=5)
    parser.add_argument("--stress_subdir", type=str, default="training_stressed")

    parser.add_argument(
        "--normal_dataset_name",
        type=str,
        default="src/baselines/DeepOPF/datasets/{case}/training_dataset",
    )
    parser.add_argument(
        "--stressed_dataset_name",
        type=str,
        default="src/baselines/DeepOPF/datasets/{case}/training_stressed_dataset",
    )

    parser.add_argument(
        "--init_model_path",
        type=str,
        default="src/baselines/DeepOPF/model/{case}/trained_model.pth",
    )
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument("--strict_load", action="store_true", default=False, 
                        help="Require the initialization checkpoint to match the model exactly when loading weights. If set, loading fails when any parameter is missing or unexpected.")

    parser.add_argument(
        "--save_model_path",
        type=str,
        default="src/baselines/DeepOPF/model/{case}/trained_model_augmented.pth",
    )

    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="Main-process PyTorch/BLAS CPU threads. 0 = auto.",
    )
    parser.add_argument(
        "--interop_threads",
        type=int,
        default=1,
        help="Main-process PyTorch inter-op threads.",
    )
    parser.add_argument(
        "--data_num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Usually 0 or small integer is enough here.",
    )
    parser.add_argument(
        "--penalty_workers",
        type=int,
        default=-1,
        help="-1 = auto, 0/1 = no multiprocessing, >1 = number of CPU worker processes for PF penalty.",
    )
    parser.add_argument("--disable_numba", action="store_true", default=False)
    parser.add_argument("--disable_lightsim2grid", action="store_true", default=False)

    args = parser.parse_args()

    if args.cpu_threads <= 0:
        args.cpu_threads = auto_main_cpu_threads()

    if args.penalty_workers == -1:
        args.penalty_workers = auto_penalty_workers()
    elif args.penalty_workers < -1:
        args.penalty_workers = 0

    normal_dataset_name = args.normal_dataset_name.format(case=args.case)
    stressed_dataset_name = args.stressed_dataset_name.format(case=args.case)
    save_model_path = args.save_model_path.format(case=args.case)

    init_model_path = ""
    if not args.train_from_scratch:
        init_model_path = args.init_model_path.format(case=args.case)

    sample_net, _ = util.get_base_net(args.case, "in_distribution")

    if not os.path.exists(normal_dataset_name + ".pt"):
        raise FileNotFoundError(f"Normal training dataset not found: {normal_dataset_name}.pt")

    X_nom, Y_nom, _, meta = load_saved_dataset(normal_dataset_name)

    if args.gen_data:
        os.makedirs(os.path.dirname(stressed_dataset_name), exist_ok=True)
        build_stressed_training_data_loader(
            base_net=sample_net,
            case=args.case,
            fname=stressed_dataset_name,
            batch_size=args.batch_size,
            scen_subdir=args.stress_subdir,
        )

    if not os.path.exists(stressed_dataset_name + ".pt"):
        raise FileNotFoundError(
            f"Stressed training dataset not found: {stressed_dataset_name}.pt. "
            f"Run once with --gen_data to create it."
        )

    X_str, Y_str, _, _ = load_saved_dataset(stressed_dataset_name)

    model = train_penalty_model_curriculum(
        base_net=sample_net,
        X_nom=X_nom,
        Y_nom=Y_nom,
        X_str=X_str,
        Y_str=Y_str,
        meta=meta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        w1=args.w1,
        w2=args.w2,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        init_model_path=init_model_path,
        strict_load=args.strict_load,
        cpu_threads=args.cpu_threads,
        interop_threads=args.interop_threads,
        data_num_workers=args.data_num_workers,
        penalty_workers=args.penalty_workers,
        enable_numba=(not args.disable_numba),
        enable_lightsim2grid=(not args.disable_lightsim2grid),
    )

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"Saved augmented model to: {save_model_path}")