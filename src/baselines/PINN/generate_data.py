'''
This file aim to generate training and test data for PINN. 
No original codebase is found for this task, hence 
this code aims to replicate the original paper as best as possible.

Author: Afia Afrin
Date: Sept, 13, 2025
'''

import os, json, copy
import numpy as np
import pandas as pd
import pandapower as pp
from src import util

# ---------------- LHS (Latin Hypercube) ----------------
'''
From paper (sec III-A): Ten thousand sets of random active and reactive power input values were 
generated using latin hypercube sampling. 
'''
def lhs_uniform(n_samples, n_dims, low=0.0, high=1.0, rng=None):
    """
    LHS: for each dimension, stratify [0,1] into n_samples bins,
    sample one point per bin, then independently permute per-dimension.
    Returns array in [low, high].
    """
    rng = np.random.default_rng() if rng is None else rng
    # Stratified samples in [0,1]
    cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
    X = np.zeros((n_samples, n_dims), dtype=float)
    for d in range(n_dims):
        X[:, d] = rng.permutation(cut)
    return low + (high - low) * X

def sample_uniform_loads(net):
    """uniformly scale loads within [0.9, 1.1].
    """
    
    rng = np.random.default_rng() 

    for idx in net.load.index:
        # Original values
        p0 = net.load.at[idx, "p_mw"]
        q0 = net.load.at[idx, "q_mvar"]

        # Sample scaling factor independently for each bus
        scale = rng.uniform(0.9, 1.1)

        # Apply perturbation
        net.load.at[idx, "p_mw"] = p0 * scale
        net.load.at[idx, "q_mvar"] = q0 * scale
    
    return net


# ---------------- helpers to collect IO ----------------
def _v_rect(vm_pu, va_deg):
    a = np.deg2rad(va_deg)
    return vm_pu*np.cos(a), vm_pu*np.sin(a)

def _collect_inputs_from_loads(n):
    cols = {}
    for li, row in n.load.iterrows():
        bus = int(row.bus)
        cols[f"P_load{li}_bus{bus}"] = float(row.p_mw)
        cols[f"Q_load{li}_bus{bus}"] = float(row.q_mvar)
    return cols

def _collect_outputs_gen_pq(n):
    cols = {}
    for gi, row in n.gen.iterrows():
        bus = int(row.bus)
        cols[f"Pg_gen{gi}_bus{bus}"] = float(n.res_gen.at[gi, "p_mw"])
        cols[f"Qg_gen{gi}_bus{bus}"] = float(n.res_gen.at[gi, "q_mvar"])
    for ei, row in n.ext_grid.iterrows():
        bus = int(row.bus)
        cols[f"Pg_slack{ei}_bus{bus}"] = float(n.res_ext_grid.at[ei, "p_mw"])
        cols[f"Qg_slack{ei}_bus{bus}"] = float(n.res_ext_grid.at[ei, "q_mvar"])
    return cols

def _collect_outputs_v(n):
    vre, vim = _v_rect(n.res_bus.vm_pu.values, n.res_bus.va_degree.values)
    cols = {}
    for i, bus in enumerate(n.bus.index.to_numpy()):
        cols[f"Vre_bus{int(bus)}"] = float(vre[i])
    for i, bus in enumerate(n.bus.index.to_numpy()):
        cols[f"Vim_bus{int(bus)}"] = float(vim[i])
    return cols

def _natural_sort(cols):
    '''
    sort the columns according to bus number (e.g. from smallest 
    to largest bus number)
    '''
    def key(c):
        b = int(c.split("bus")[1]) if "bus" in c else 0
        head = c.split("_")[0]
        return (head, b, c)
    return sorted(cols, key=key)

def _save_norm_stats(df, path_json):
    stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0) or 1.0)}
             for c in df.columns}
    with open(path_json, "w") as f:
        json.dump(stats, f, indent=2)

def _apply_norm(df, stats_json_path):
    with open(stats_json_path, "r") as f:
        stats = json.load(f)
    df_n = df.copy()
    for c in df.columns:
        mu = stats[c]["mean"]; sd = stats[c]["std"]
        df_n[c] = (df[c] - mu) / (sd if sd != 0 else 1.0)
    return df_n

def run_opf(net):
    # Prefer PowerModels (IPOPT) if available; else fallback to pandapower’s OPF
    try:
        pp.runpm_ac_opf(net, pm_solver="ipopt")
    except Exception:
        pp.runopp(net)


def generate_training_data(
    num_bus,
    case
):
    """
    TRAIN set using uniform per-load perturbations (sample_uniform_loads):
      - For each sample, deepcopy base net, uniformly scale each load row in [0.9, 1.1]
      - Run OPF; if not converged, skip sample (print a message) and continue
      - Keep going until we have n_total/2 feasible samples (paper’s “data” half)
      - Save normalized CSVs + training stats
    """
    out_dir="src/baselines/PINN/datasets/" + num_bus + "/"
    os.makedirs(out_dir, exist_ok=True)

    # slightly diff. thatn paper: 50% data, 50% collocation
    rows_in, rows_pq, rows_v = [], [], []
    scen_dir= "src/scenarios/" + case + "/training"
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]

    for path in paths:
        #load the pre-saved net
        ncopy = util.load_net(path)
    
        try:
            run_opf(ncopy)
            if not getattr(ncopy, "OPF_converged", False):
                print(f"OPF not converged (train). Skipping this sample.")
                continue
        except Exception as e:
            print(f"OPF error (train): {e}. Skipping this sample.")
            continue

        rows_in.append(_collect_inputs_from_loads(ncopy))
        rows_pq.append(_collect_outputs_gen_pq(ncopy))
        rows_v.append(_collect_outputs_v(ncopy))
        #kept += 1
        #tries += 1

    # ---- pack & normalize ----
    df_in = pd.DataFrame(rows_in).reindex(_natural_sort(pd.DataFrame(rows_in).columns), axis=1)
    df_pq = pd.DataFrame(rows_pq).reindex(_natural_sort(pd.DataFrame(rows_pq).columns), axis=1)
    df_v  = pd.DataFrame(rows_v ).reindex(_natural_sort(pd.DataFrame(rows_v ).columns), axis=1)

    stats_in = os.path.join(out_dir, "norm_stats.json")
    stats_pq = os.path.join(out_dir, "norm_stats_pq.json")
    stats_v  = os.path.join(out_dir, "norm_stats_v.json")
    _save_norm_stats(df_in, stats_in)
    _save_norm_stats(df_pq, stats_pq)
    _save_norm_stats(df_v,  stats_v)

    df_in_n = _apply_norm(df_in, stats_in)
    df_pq_n = _apply_norm(df_pq, stats_pq)
    df_v_n  = _apply_norm(df_v,  stats_v)

    f_in_n = os.path.join(out_dir, "NN_input_train.csv")
    f_pq_n = os.path.join(out_dir, "NN_output_PQ_train.csv")
    f_v_n  = os.path.join(out_dir, "NN_output_V_train.csv")
    df_in_n.to_csv(f_in_n, index=False)
    df_pq_n.to_csv(f_pq_n, index=False)
    df_v_n.to_csv(f_v_n,  index=False)

    #return {"train_datasets": (f_in_n, f_pq_n, f_v_n), "stats": (stats_in, stats_pq, stats_v)}


def generate_test_data(num_bus, case, which_data):
    """
    TEST set using uniform per-load perturbations (sample_uniform_loads):
      - For each sample, deepcopy base net, uniformly scale each load row in [0.9, 1.1]
      - Run OPF; if not converged, skip sample (print a message) and continue
      - Keep going until we have n_test feasible samples
      - Normalize using *training* stats in stats_dir
    """
    out_dir="src/baselines/PINN/datasets/" + num_bus + "/"
    stats_dir="src/baselines/PINN/datasets/" + num_bus + "/"
    os.makedirs(out_dir, exist_ok=True)

    rows_in, rows_pq, rows_v = [], [], []

    scen_dir= "src/scenarios/" + case + "/test/" + which_data + "/"
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]

    #nets_list= []
    for path in paths:
        #load the pre-saved net
        ncopy = util.load_net(path)
        try:
            run_opf(ncopy)
            if not getattr(ncopy, "OPF_converged", False):
                print("OPF not converged (test). Skipping this sample.")
                 
                continue
        except Exception as e:
            print(f"OPF error (test): {e}. Skipping this sample.")
             
            continue

        #nets_list.append(path)   
        rows_in.append(_collect_inputs_from_loads(ncopy))
        rows_pq.append(_collect_outputs_gen_pq(ncopy))
        rows_v.append(_collect_outputs_v(ncopy))
         

    # ---- pack & normalize with TRAIN stats ----
    df_in = pd.DataFrame(rows_in).reindex(_natural_sort(pd.DataFrame(rows_in).columns), axis=1)
    df_pq = pd.DataFrame(rows_pq).reindex(_natural_sort(pd.DataFrame(rows_pq).columns), axis=1)
    df_v  = pd.DataFrame(rows_v ).reindex(_natural_sort(pd.DataFrame(rows_v ).columns), axis=1)

    stats_in = os.path.join(stats_dir, "norm_stats.json")
    stats_pq = os.path.join(stats_dir, "norm_stats_pq.json")
    stats_v  = os.path.join(stats_dir, "norm_stats_v.json")

    df_in_n = _apply_norm(df_in, stats_in)
    df_pq_n = _apply_norm(df_pq, stats_pq)
    df_v_n  = _apply_norm(df_v,  stats_v)

    f_in_n = os.path.join(out_dir, "NN_input_test_" + which_data +".csv")
    f_pq_n = os.path.join(out_dir, "NN_output_PQ_test_" + which_data +".csv")
    f_v_n  = os.path.join(out_dir, "NN_output_V_test_" + which_data +".csv")
    df_in_n.to_csv(f_in_n, index=False)
    df_pq_n.to_csv(f_pq_n, index=False)
    df_v_n.to_csv(f_v_n,  index=False)

    # nets_list_path= os.path.join(out_dir,'nets_path_list.json')
    # with open(nets_list_path, 'w') as f:
    #     json.dump(nets_list, f )

    #return {"test_datasets": (f_in_n, f_pq_n, f_v_n)}
