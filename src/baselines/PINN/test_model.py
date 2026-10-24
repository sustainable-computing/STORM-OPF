# src/baselines/PINN/test_model.py
import os, re, json, copy, argparse, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorflow import keras
from src import util, view, global_vars as gvars
from src.global_evaluation_AC import PFEvaluator
from src.baselines.PINN import generate_data

#helper 
def prep_constraints(net):
    '''
    in case base_net doesn't have all constraints associated, 
    use this method to make sure it adds the constraints
    '''
    # voltage defaults
    if "min_vm_pu" not in net.bus: net.bus["min_vm_pu"] = 0.95
    if "max_vm_pu" not in net.bus: net.bus["max_vm_pu"] = 1.05

    # gen P/Q defaults if missing
    for c in ["min_p_mw","max_p_mw","min_q_mvar","max_q_mvar"]:
        if c not in net.gen: net.gen[c] = np.nan
    net.gen["min_p_mw"] = net.gen["min_p_mw"].fillna(0.0)
    net.gen["max_p_mw"] = net.gen["max_p_mw"].fillna(np.maximum(1.0, net.gen["p_mw"]*1.2))
    net.gen["min_q_mvar"] = net.gen["min_q_mvar"].fillna(-1e3)
    net.gen["max_q_mvar"] = net.gen["max_q_mvar"].fillna(+1e3)

    # line/trafo ratings
    if len(net.line):
        if "max_loading_percent" not in net.line: net.line["max_loading_percent"] = 100.0
        net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(100.0)
    if len(net.trafo):
        if "max_loading_percent" not in net.trafo: net.trafo["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = net.trafo["max_loading_percent"].fillna(100.0)
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        if "max_loading_percent" not in net.trafo3w: net.trafo3w["max_loading_percent"] = 100.0
        net.trafo3w["max_loading_percent"] = net.trafo3w["max_loading_percent"].fillna(100.0)


# -----------------------------
# Helpers for stats / columns
# -----------------------------
def _load_stats_as_arrays(stats_path: str, col_order):
    """
    stats_path: JSON with {"col": {"mean":..., "std":...}, ...}
    Returns (mu, sd) arrays ordered to match col_order.
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mu = np.array([stats[c]["mean"] for c in col_order], dtype=float)
    sd = np.array([stats[c]["std"] if stats[c]["std"] != 0 else 1.0 for c in col_order], dtype=float)
    return mu, sd


def _parse_bus_from_Vcol(col):
    # "Vre_bus7" or "Vim_bus12" -> 7 / 12
    m = re.search(r"bus(\d+)$", col)
    return int(m.group(1)) if m else None


def _parse_load_col(col):
    # "P_load3_bus14" / "Q_load3_bus14" -> kind='P'/'Q', load_idx=3, bus=14
    m = re.match(r"([PQ])_load(\d+)_bus(\d+)$", col)
    if not m: return None
    kind, li, bus = m.group(1), int(m.group(2)), int(m.group(3))
    return kind, li, bus


def _parse_pq_col(col):
    # "Pg_gen4_bus2", "Qg_gen4_bus2", "Pg_slack0_bus1", "Qg_slack0_bus1"
    m_gen = re.match(r"([PQ]g)_gen(\d+)_bus(\d+)$", col)
    if m_gen:
        pqg, gi, bus = m_gen.group(1), int(m_gen.group(2)), int(m_gen.group(3))
        return ("gen", pqg, gi, bus)
    m_slk = re.match(r"([PQ]g)_slack(\d+)_bus(\d+)$", col)
    if m_slk:
        pqg, ei, bus = m_slk.group(1), int(m_slk.group(2)), int(m_slk.group(3))
        return ("slack", pqg, ei, bus)
    return None


# -----------------------------
# PINN-specific adapters
# -----------------------------
def pinn_denorm_outputs(Y_norm, meta, device="cpu"):
    """
    Split Y_norm into [V_norm | PQ_norm | (ignore rest...)] and de-normalize
    using training stats stored in meta.
    Returns (V_denorm, PQ_denorm) as torch tensors on `device`.
    """
    if not torch.is_tensor(Y_norm):
        Y_norm = torch.as_tensor(Y_norm, dtype=torch.float32, device=device)

    Vdim = meta["V_dim"]
    PQdim = meta["PQ_dim"]

    Vn = Y_norm[:, :Vdim]
    PQn = Y_norm[:, Vdim:Vdim+PQdim]

    V_mu= torch.as_tensor(meta["V_mu"],  dtype=Vn.dtype, device=device).view(1, -1)
    V_sd = torch.as_tensor(meta["V_sd"],  dtype=Vn.dtype, device=device).view(1, -1)
    PQ_mu= torch.as_tensor(meta["PQ_mu"], dtype=Vn.dtype, device=device).view(1, -1)
    PQ_sd= torch.as_tensor(meta["PQ_sd"], dtype=Vn.dtype, device=device).view(1, -1)

    V= V_mu  + Vn  * V_sd
    PQ = PQ_mu + PQn * PQ_sd
    return V, PQ


def pinn_apply_prediction_to_net(base_net, meta, x_norm_i, y_norm_i, denorm_fn, device="cpu"):
    """
    - Denormalize inputs (loads) using meta["IN_mu/sd"] and write to net.load rows.
    - Denormalize model outputs using denorm_fn -> (V, PQ).
    - From Vre/Vim, compute |V| at each bus; set:
        * generator vm_pu to |V| at their buses (PV buses),
        * slack ext_grid vm_pu to |V| at slack bus(es)
    - From PQ, set generator active power P (Pg_gen*) at each gen row.
      (We do NOT fix Q; PF will solve Q subject to limits.)
    """
    
    net = copy.deepcopy(base_net)

    # 1) Inputs: set net.load p/q from de-normalized features
    x = x_norm_i.detach().cpu().numpy() if hasattr(x_norm_i, "detach") else np.asarray(x_norm_i)
    IN_mu, IN_sd, IN_cols = meta["IN_mu"], meta["IN_sd"], meta["IN_cols"]
    xin = IN_mu + x * IN_sd
    for j, col in enumerate(IN_cols):
        parsed = _parse_load_col(col)
        if not parsed: continue
        kind, li, _bus = parsed
        if kind == "P":
            if li in net.load.index:
                net.load.at[li, "p_mw"] = float(xin[j])
        else:
            if li in net.load.index:
                net.load.at[li, "q_mvar"] = float(xin[j])

    # 2) Outputs: denorm V & PQ
    V_den, PQ_den = denorm_fn(y_norm_i.view(1, -1) if hasattr(y_norm_i, "view") else np.asarray(y_norm_i).reshape(1, -1),
                              meta, device=device)
    V_den = V_den.squeeze(0).detach().cpu().numpy()
    PQ_den = PQ_den.squeeze(0).detach().cpu().numpy()

    # 2a) Turn Vre/Vim into |V| per bus
    V_cols = meta["V_cols"]
    vre_map, vim_map = {}, {}
    for k, c in enumerate(V_cols):
        if c.startswith("Vre_"):
            b = _parse_bus_from_Vcol(c);  vre_map[b] = V_den[k]
        elif c.startswith("Vim_"):
            b = _parse_bus_from_Vcol(c);  vim_map[b] = V_den[k]
    vm_by_bus = {}
    for b in net.bus.index.to_numpy():
        vre = float(vre_map.get(int(b), 0.0))
        vim = float(vim_map.get(int(b), 0.0))
        vm_by_bus[int(b)] = float(np.hypot(vre, vim))

    # 2b) Pull Pg predictions for each gen; also capture |V| for slack/gen buses
    PQ_cols = meta["PQ_cols"]
    Pg_gen_pred = {}     # gi -> Pg value
    #Vm_gen_pred = {}     # bus -> |V|
    #Vm_slack_pred = {}   # slack row id -> |V| (set on its bus)

    for k, c in enumerate(PQ_cols):
        parsed = _parse_pq_col(c)
        if not parsed: continue
        kind, pqg, row_id, bus = parsed
        if kind == "gen" and pqg == "Pg":
            Pg_gen_pred[row_id] = float(PQ_den[k])

    # Write generator P_mw
    for gi, pval in Pg_gen_pred.items():
        if gi in net.gen.index:
            net.gen.at[gi, "p_mw"] = pval

    # Set PV generator bus |V| targets from predicted |V|
    for gi in net.gen.index:
        bus_i = int(net.gen.at[gi, "bus"])
        vm = vm_by_bus.get(bus_i, None)
        if vm is not None:
            net.gen.at[gi, "vm_pu"] = vm

    # Set slack ext_grid vm_pu from predicted |V| at its buses
    for ei in net.ext_grid.index:
        bus_e = int(net.ext_grid.at[ei, "bus"])
        vm = vm_by_bus.get(bus_e, None)
        if vm is not None:
            net.ext_grid.at[ei, "vm_pu"] = vm

    # Final consistency tweaks (limits, types, etc.)
    prep_constraints(net)
    return net

def model_infer(model, xbatch, device='cpu'):
    """
    Convert a PyTorch batch to NumPy, supply Keras with [L_Val, L_type],
    get heads [V, PQ, ...], and return a torch Tensor concatenated as [V | PQ].
    """

    # xbatch comes from a PyTorch DataLoader: shape (B, in_dim)
    if torch.is_tensor(xbatch):
        X_np = xbatch.detach().cpu().numpy().astype(np.float32)
    elif isinstance(xbatch, np.ndarray):
        X_np = xbatch.astype(np.float32)
    else:
        # If somehow a list/tuple arrives, try to stack
        X_np = np.asarray(xbatch, dtype=np.float32)

    B = X_np.shape[0]
    # PINN input #2 used during training was a ones column (L_type)
    L_type = np.ones((B, 1), dtype=np.float32)

    # Keras predict -> list of heads [V, PQ, ...]
    preds = model.predict([X_np, L_type], verbose=0)

    if not isinstance(preds, (list, tuple)) or len(preds) < 2:
        raise RuntimeError("Expected Keras model to output [V, PQ, ...]. Got something else.")

    V_head  = preds[0]  # (B, V_dim)
    PQ_head = preds[1]  # (B, PQ_dim)

    # IMPORTANT: return [V | PQ] to match pinn_denorm_outputs slicing
    Y_np = np.concatenate([V_head, PQ_head], axis=1).astype(np.float32)

    # Return a torch tensor (CPU) so downstream code treats it like other models
    return torch.from_numpy(Y_np)

# -----------------------------
# Simple CSV loader for PINN
# -----------------------------
def _build_pinn_test_loader(num_bus_str: str, which_data: str, batch: int):
    base = os.path.join("src", "baselines", "PINN", "datasets", num_bus_str)

    fX = os.path.join(base, f"NN_input_test_{which_data}.csv")
    if not os.path.exists(fX):
        raise FileNotFoundError(f"Missing test inputs: {fX}")

    dfX = pd.read_csv(fX)
    X = torch.as_tensor(dfX.to_numpy(), dtype=torch.float32)
    # Dummy Y to match evaluator's shape 
    Y = torch.zeros((X.shape[0], 1), dtype=torch.float32)

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    return loader, list(dfX.columns)


def _load_meta( num_bus_str: str, which_data: str):
    base = os.path.join("src", "baselines", "PINN", "datasets", num_bus_str)

    # Columns (order) come from the saved test CSVs
    fV = os.path.join(base, f"NN_output_V_test_{which_data}.csv")
    fPQ = os.path.join(base, f"NN_output_PQ_test_{which_data}.csv")
    dfV= pd.read_csv(fV)
    dfPQ = pd.read_csv(fPQ)

    # Training stats
    stats_in = os.path.join(base, "norm_stats.json")
    stats_pq = os.path.join(base, "norm_stats_pq.json")
    stats_v = os.path.join(base, "norm_stats_v.json")

    IN_mu, IN_sd = _load_stats_as_arrays(stats_in, list(pd.read_csv(os.path.join(base, f"NN_input_test_{which_data}.csv")).columns))
    PQ_mu, PQ_sd = _load_stats_as_arrays(stats_pq, list(dfPQ.columns))
    V_mu,  V_sd  = _load_stats_as_arrays(stats_v,  list(dfV.columns))

    meta = {
        "IN_cols": list(pd.read_csv(os.path.join(base, f"NN_input_test_{which_data}.csv")).columns),
        "IN_mu": IN_mu,
        "IN_sd": IN_sd,

        "PQ_cols": list(dfPQ.columns),
        "PQ_mu": PQ_mu,
        "PQ_sd": PQ_sd,
        "PQ_dim": dfPQ.shape[1],

        "V_cols": list(dfV.columns),
        "V_mu": V_mu,
        "V_sd": V_sd,
        "V_dim": dfV.shape[1],
    }
    return meta


def _load_test_net_paths(case: str, which_data: str):
    scen_dir = os.path.join("src", "scenarios", case, "test", which_data)
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    return [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--gen_data", action="store_true", default=False)
    parser.add_argument("--which_data", type=str, default="in_distribution")
    parser.add_argument("--which_model", type=str, default='baseline',
                        help='baseline or augmented? Which one to test?')
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    
    # Base net & counts
    sample_net, _ = util.get_base_net(args.case, "in_distribution")  # base nets are always in_dist
    num_bus = len(sample_net.bus)
    num_bus_str = str(num_bus)
    #gen data?
    if args.gen_data:
        generate_data.generate_test_data(num_bus_str, args.case, args.which_data)
    # Loader + meta + aligned net paths
    loader, in_cols = _build_pinn_test_loader(num_bus_str, args.which_data, batch=args.batch)
    meta = _load_meta( num_bus_str, args.which_data)
    net_paths = _load_test_net_paths(args.case, args.which_data)
    #load trained model 
    if(args.which_model == 'baseline'): 
        model_dir = os.path.join("src", "baselines", "PINN", "trained_model", num_bus_str)
        print('Loaded BASELINE model')
    elif (args.which_model == 'augmented'): 
        model_dir = os.path.join("src", "baselines", "PINN", "trained_model_augmented", num_bus_str)
        print('Loaded AUGMENTED model')
    model = keras.models.load_model(model_dir)

    print("Running PINN evaluation")
    evaluator = PFEvaluator(
        denorm_fn=pinn_denorm_outputs,
        apply_pred_fn=pinn_apply_prediction_to_net,
        model_infer_fn= model_infer,
        feas_tol=gvars.feas_tol,
        max_penalty=gvars.max_pen,
        ctol=gvars.ctol,
        runpp_kwargs=dict(enforce_q_lims=True, calculate_voltage_angles=True),
    )

    pinn_res = evaluator.evaluate(
        model=model,
        loader=loader,
        meta=meta,
        net_paths=net_paths,
        device=str(device),
        collect_norm_mse=False,  # we don’t use Y here
        verbose=False,
    )

    print(pinn_res["feasibility_rate_%"])

    if(args.which_model == 'baseline'):fname = os.path.join('results', args.case, f"PINN_result_{args.which_data}.pkl")
    elif (args.which_model == 'augmented'):fname = os.path.join('results', args.case, f"PINN_augmented_result_{args.which_data}.pkl")
    
    with open(fname, "wb") as f:
        pickle.dump(pinn_res, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved PINN results to {fname}")
    

if __name__ == "__main__":
    main()
