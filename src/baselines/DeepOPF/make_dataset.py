import os 
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from src import global_vars as gvars,  util

'''
IMPORTANT: do not use any seeds here, we don't want reproducibility 
for this task. We want different dataset to be generated in 
different runs.
'''

def aggregate_bus_vector(net, series_name):
    """
    Returns a vector (len = n_buses) with per-bus sum of a load series (p_mw or q_mvar).
    """
    n_b = len(net.bus)
    vec = np.zeros(n_b, dtype=float)
    if len(net.load):
        # Map load rows to bus indices (0-based)
        for _, row in net.load.iterrows():
            bus_idx = int(row['bus'])
            vec[bus_idx] += float(row[series_name])
    return vec

# ------------- main builder -------------
def build_deepopf_training_data_loader(base_net, case, fname, batch_size=128):
    '''
    PARAMS:
    base_net: a sample net from which we can obtain the network topology info
    case: which test case (14/18/118...)
    fname: file save name 
    '''
    
    # Index sets; mirror DeepOPF roles
    slack_bus = int(base_net.ext_grid.bus.iloc[0])             # single slack assumed
    pv_bus_list = [int(b) for b in base_net.gen.bus.values]    # buses with generators (PV buses)
    pv_bus_list = [b for b in pv_bus_list if b != slack_bus]   # exclude slack bus

    # Active generators = net.gen rows (exclude any out-of-service; exclude slack side)
    active_gen_mask = (base_net.gen.in_service.values.astype(bool))
    active_gen_bus = base_net.gen.bus.values
    active_gen_idx = [i for i, (m, b) in enumerate(zip(active_gen_mask, active_gen_bus))
                      if m and int(b) != slack_bus]

    # Per-gen P bounds (for normalization like DeepOPF’s PG_Lowbound/Upbound)
    # If bounds missing, infer a small band around current p as a fallback.
    pmin = np.zeros(len(active_gen_idx), dtype=float)
    pmax = np.zeros(len(active_gen_idx), dtype=float)
    for k, gi in enumerate(active_gen_idx):
        lo = base_net.gen.at[gi, 'min_p_mw'] if 'min_p_mw' in base_net.gen.columns else None
        hi = base_net.gen.at[gi, 'max_p_mw'] if 'max_p_mw' in base_net.gen.columns else None
        if lo is None or np.isnan(lo): lo = 0.0
        if hi is None or np.isnan(hi): hi = max(1.0, float(base_net.gen.at[gi, 'p_mw']) * 1.2)
        pmin[k], pmax[k] = lo, max(hi, lo + 1e-3)

    # Voltage bounds (bus-wise) & slack voltage bounds (use net defaults if present)
    # pandapower buses usually carry min_vm_pu/max_vm_pu (default 0.95/1.05)
    vm_min = base_net.bus.get('min_vm_pu', pd.Series(0.95, index=base_net.bus.index)).values
    vm_max = base_net.bus.get('max_vm_pu', pd.Series(1.05, index=base_net.bus.index)).values
    vm_min = np.asarray(vm_min, dtype=float)
    vm_max = np.asarray(vm_max, dtype=float)

    slack_v_min = float(vm_min[slack_bus])
    slack_v_max = float(vm_max[slack_bus])

    # Collect samples
    P_D_list, Q_D_list = [], []
    PG_active_list, Vm_list, SlackV_list = [], [], []

    scen_dir= "src/scenarios/" + case + "/training"
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]

    for path in paths:
        #load the pre-saved net
        net = util.load_net(path)
        # Inputs: per-bus P_D, Q_D
        P_D_vec = aggregate_bus_vector(net, 'p_mw')   # shape (n_buses,)
        Q_D_vec = aggregate_bus_vector(net, 'q_mvar') # shape (n_buses,)

        # Outputs:
        # 1) Active PG (aligned with active_gen_idx)
        # net.res_gen contains only gen (not ext_grid). Indexing by row index is OK.
        PG_vec = np.array([float(net.res_gen.at[gi, 'p_mw']) for gi in active_gen_idx], dtype=float)

        # 2) Slack V magnitude
        slack_v = float(net.res_bus.at[slack_bus, 'vm_pu'])

        # 3) Bus Vm at PV buses (i.e. gen buses) (exclude slack)
        Vm = net.res_bus.vm_pu.values.astype(float)
        Vm_pv = Vm[pv_bus_list]
        # Append
        P_D_list.append(P_D_vec)
        Q_D_list.append(Q_D_vec)
        PG_active_list.append(PG_vec)
        SlackV_list.append(slack_v)
        Vm_list.append(Vm_pv)

    # Stack
    P_D = np.vstack(P_D_list) if P_D_list else np.empty((0, len(base_net.bus)))
    Q_D = np.vstack(Q_D_list) if Q_D_list else np.empty((0, len(base_net.bus)))
    PG_active = np.vstack(PG_active_list) if PG_active_list else np.empty((0, len(active_gen_idx)))
    SlackV = np.asarray(SlackV_list, dtype=float).reshape(-1, 1)
    Vm_pv = np.vstack(Vm_list) if Vm_list else np.empty((0, len(pv_bus_list)))

    # -------- match DeepOPF normalizations --------
    # Inputs (z-score per feature)
    P_mu, P_sig = P_D.mean(axis=0), P_D.std(axis=0) + 1e-8
    Q_mu, Q_sig = Q_D.mean(axis=0), Q_D.std(axis=0) + 1e-8
    X = np.concatenate([(P_D - P_mu)/P_sig, (Q_D - Q_mu)/Q_sig], axis=1)

    # Outputs:
    # 1) PG_active -> min-max per gen
    PG_lo = pmin[None, :]
    PG_hi = pmax[None, :]
    PG_norm = (PG_active - PG_lo) / (PG_hi - PG_lo + gvars.eps)

    # 2) Slack V -> min-max
    Slack_lo = slack_v_min
    Slack_hi = slack_v_max
    Slack_norm = (SlackV - Slack_lo) / (Slack_hi - Slack_lo + gvars.eps)

    # 3) Vm at PV buses -> min-max per bus
    Vm_lo = vm_min[pv_bus_list][None, :]
    Vm_hi = vm_max[pv_bus_list][None, :]
    Vm_norm = (Vm_pv - Vm_lo) / (Vm_hi - Vm_lo + gvars.eps)

    Y = np.concatenate([PG_norm, Slack_norm, Vm_norm], axis=1)

    # Torch tensors 
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    artifacts = { 
        "P_mu": P_mu, "P_sig": P_sig, "Q_mu": Q_mu, "Q_sig": Q_sig,
        "active_gen_idx": active_gen_idx,
        "pv_bus_list": pv_bus_list,
        "PG_lo": pmin, "PG_hi": pmax,
        "slack_bus": slack_bus, "Slack_lo": slack_v_min, "Slack_hi": slack_v_max,
        "Vm_lo": vm_min[pv_bus_list], "Vm_hi": vm_max[pv_bus_list],
    }
    torch.save(
        {"dataset": dataset, "meta": artifacts}, fname +".pt"
    )
    return loader, artifacts


def build_deepopf_test_data_loader(
    numTotalBus,
    case, 
    fname,
    which_data,
    batch_size=256,  
):
    """
    Generates ground truth test data for either in-distribution setup or 
    edge cases. The base_net is either a normal one or a stressed one.
    - Uses training normalization stats:
        X = zscore with training P_mu/P_sig, Q_mu/Q_sig
        Y = min-max with training bounds PG_lo/hi, Slack_lo/hi, Vm_lo/hi

    Returns:
        loader, artifacts
      where artifacts includes masks, feasibility flags, penalties, costs, timing, etc.
    """

    #load training meta
    meta_path= "src/baselines/DeepOPF/datasets/" + case +"/training_dataset.pt"
    loaded = torch.load(meta_path, map_location="cpu")
    train_meta = loaded["meta"]

    # ----- unpack training stats/bounds (NO recomputation ) -----
    P_mu = np.asarray(train_meta["P_mu"], dtype=float)
    P_sig = np.asarray(train_meta["P_sig"], dtype=float)
    Q_mu = np.asarray(train_meta["Q_mu"], dtype=float)
    Q_sig = np.asarray(train_meta["Q_sig"], dtype=float)

    PG_lo_train = np.asarray(train_meta["PG_lo"], dtype=float)          # shape (n_active_gen,)
    PG_hi_train = np.asarray(train_meta["PG_hi"], dtype=float)
    Vm_lo_train = np.asarray(train_meta["Vm_lo"], dtype=float)          # shape (n_pv_bus,)
    Vm_hi_train = np.asarray(train_meta["Vm_hi"], dtype=float)

    slack_bus = int(train_meta["slack_bus"])
    Slack_lo_train = float(train_meta["Slack_lo"])
    Slack_hi_train = float(train_meta["Slack_hi"])

    active_gen_idx = list(train_meta["active_gen_idx"])
    pv_bus_list = list(train_meta["pv_bus_list"])

    # ----- collectors -----
    P_D_list, Q_D_list = [], []
    PG_active_list, Vm_list, SlackV_list = [], [], []

    # ----- scenario loop -----
    scen_dir= "src/scenarios/" + case + "/test/" + which_data
    manifest = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    paths = [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]
    
    for path in paths:
        net = util.load_net (path)
        # Inputs (always defined)
        P_D_vec = aggregate_bus_vector(net, 'p_mw')   # shape (n_buses,)
        Q_D_vec = aggregate_bus_vector(net, 'q_mvar') # shape (n_buses,)

        PG_vec = np.full((len(active_gen_idx),), np.nan, dtype=float)
        slack_v = np.nan
        Vm_pv = np.full((len(pv_bus_list),), np.nan, dtype=float)

        PG_vec = np.array([float(net.res_gen.at[gi, 'p_mw']) for gi in active_gen_idx], dtype=float)
        slack_v = float(net.res_bus.at[slack_bus, 'vm_pu'])
        Vm = net.res_bus.vm_pu.values.astype(float)
        Vm_pv = Vm[pv_bus_list]

        # Append I/O
        P_D_list.append(P_D_vec)
        Q_D_list.append(Q_D_vec)
        PG_active_list.append(PG_vec)
        SlackV_list.append(slack_v)
        Vm_list.append(Vm_pv)
        

    # ----- stack -----
    P_D = np.vstack(P_D_list) if P_D_list else np.empty((0, numTotalBus))
    Q_D = np.vstack(Q_D_list) if Q_D_list else np.empty((0, numTotalBus))
    PG_active = np.vstack(PG_active_list) if PG_active_list else np.empty((0, len(active_gen_idx)))
    SlackV = np.asarray(SlackV_list, dtype=float).reshape(-1, 1)
    Vm_pv = np.vstack(Vm_list) if Vm_list else np.empty((0, len(pv_bus_list)))
    
    # ----- TEST-TIME NORMALIZATION (using training mu, sigma for NO LEAKAGE) -----
    # Inputs: z-score with *training* stats
    X_left = (P_D - P_mu) / (P_sig + 1e-8)
    X_right = (Q_D - Q_mu) / (Q_sig + 1e-8)
    X = np.concatenate([X_left, X_right], axis=1)

    # Outputs: min-max with *training* bounds
    PG_lo = PG_lo_train[None, :]
    PG_hi = PG_hi_train[None, :]
    PG_norm = (PG_active - PG_lo) / (PG_hi - PG_lo + gvars.eps)

    Slack_lo = Slack_lo_train
    Slack_hi = Slack_hi_train
    Slack_norm = (SlackV - Slack_lo) / (Slack_hi - Slack_lo + gvars.eps)

    Vm_lo = Vm_lo_train[None, :]
    Vm_hi = Vm_hi_train[None, :]
    Vm_norm = (Vm_pv - Vm_lo) / (Vm_hi - Vm_lo + gvars.eps)

    Y = np.concatenate([PG_norm, Slack_norm, Vm_norm], axis=1)

    # ----- torch tensors -----
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    
    # Package into a dataset
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #everything necessary for denormalization & evaluation later
    #are packed inside artifacts/meta
    artifacts = {
        "which_data": which_data,
        "P_mu": P_mu, "P_sig": P_sig, "Q_mu": Q_mu, "Q_sig": Q_sig,  # copied from training 
        "active_gen_idx": active_gen_idx,
        "pv_bus_list": pv_bus_list,
        "PG_lo": PG_lo_train, "PG_hi": PG_hi_train,
        "slack_bus": slack_bus, "Slack_lo": Slack_lo_train, "Slack_hi": Slack_hi_train,
        "Vm_lo": Vm_lo_train, "Vm_hi": Vm_hi_train,
    }

    torch.save({"dataset": dataset, "meta": artifacts}, fname + ".pt")
    return loader, artifacts