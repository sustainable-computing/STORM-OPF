import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from src import util
from src.baselines.DeepOPF.make_dataset import build_deepopf_training_data_loader
import pandapower as pp
import argparse 
from src.baselines.DeepOPF import vars 
from src import global_vars as gvars

torch.manual_seed(vars.training_seed); np.random.seed(vars.training_seed)
# -----------------------------
# helpers
# -----------------------------
def prep_constraints(net):
    '''
    in case my base_net doesn't have all constraints associated, 
    use this method to make sure it adds the constraints
    '''
    # voltage defaults
    if "min_vm_pu" not in net.bus: net.bus["min_vm_pu"] = 0.95
    if "max_vm_pu" not in net.bus: net.bus["max_vm_pu"] = 1.05

    # gen P/Q defaults if missing
    for c in ["min_p_mw","max_p_mw","min_q_mvar","max_q_mvar"]:
        if c not in net.gen: net.gen[c] = np.nan
    net.gen["min_p_mw"]   = net.gen["min_p_mw"].fillna(0.0)
    net.gen["max_p_mw"]   = net.gen["max_p_mw"].fillna(np.maximum(1.0, net.gen["p_mw"]*1.2))
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

def build_hist_initialization(base_net, meta, Y_t):
    """
    Build initial guesses for Newton PF from training-set averages.
    - Slack Vm: mean over labels (denormalized)
    - PV-bus Vm: mean per PV bus (denormalized)
    - PQ-bus Vm: 1.0 pu (reasonable default)
    - All bus angles Va: 0.0 deg (robust default)
    """
    n_bus = len(base_net.bus)
    pv_bus_list = meta["pv_bus_list"]
    slack_bus = int(meta["slack_bus"])

    n_pg = len(meta["active_gen_idx"])
    n_pv = len(pv_bus_list)

    # Y_t = [PG_active_norm | SlackV_norm | Vm_PV_norm]
    slack_norm = Y_t[:, n_pg]                                # (N,)
    vm_pv_norm = Y_t[:, n_pg+1 : n_pg+1+n_pv]                # (N, n_pv)

    Slack_lo, Slack_hi = meta["Slack_lo"], meta["Slack_hi"]  # scalars
    Vm_lo, Vm_hi = meta["Vm_lo"], meta["Vm_hi"]              # arrays (n_pv,)

    SlackV_mean = float(Slack_lo + slack_norm.mean().item() * (Slack_hi - Slack_lo))
    Vm_PV_mean = (Vm_lo + vm_pv_norm.mean(dim=0).cpu().numpy() * (Vm_hi - Vm_lo)).ravel()

    vm_init = np.ones(n_bus, dtype=float) * 1.0
    vm_init[slack_bus] = SlackV_mean
    for i, b in enumerate(pv_bus_list):
        vm_init[int(b)] = float(Vm_PV_mean[i])

    va_init = np.zeros(n_bus, dtype=float)  # 0° for all buses

    return vm_init, va_init

def run_pf(net, vm_init=None, va_init=None):
    # DeepOPF-like: do NOT enforce Q limits inside PF
    kwargs = dict(
        enforce_q_lims=False,  #was false, jan 16               
        calculate_voltage_angles=True,
    )
    if vm_init is not None or va_init is not None:
        pp.runpp(
            net,
            init="auto",
            init_vm_pu=vm_init,
            init_va_degree=va_init,
            **kwargs
        )
    else:
        pp.runpp(net, **kwargs)


def compute_penalty_from_net(net, ctol= gvars.ctol) -> float:
    """
    Computes a single scalar "constraint-violation penalty" from a solved pandapower network.
    Functionality is similar to the `zero_order_penalty_abs` method in original DeepOPF paper.
    This aggregates violations from (i) generator P/Q limits, (ii) PQ-bus voltage limits,
    and (iii) thermal loading limits on lines and transformers. Each violation term is
    nonnegative and **scaled by the system MVA base** (`net.sn_mva`) so the final penalty
    is roughly in “per-base-MVA” units (e.g., MW / baseMVA).

    --------------
    call this **after** a successful power flow or OPF solve, so that
    `net.res_*` result tables are populated:
        >>> pp.runpp(net, enforce_q_lims=True, calculate_voltage_angles=True)
        >>> pen = compute_penalty_from_net(net)
        >>> is_feasible = (pen <= 1e-6)  # typical tolerance

    Assumptions & Notes
    -------------------
    - The network has been solved so that `net.res_*` tables exist. If PF/OPF fails,
    call sites should catch exceptions and assign a large penalty (e.g., 1e9).
    - If constraints (e.g., gen bounds, voltage limits) are missing, this function
    will either skip or fall back to simple defaults (voltage 0.95 - 1.05 p.u.).
    - For lines/transformers, the penalty relies on `loading_percent` reported by
    pandapower results; if no rating information is available, the term contributes 0.

    Parameters
    ----------
    net : pandapowerNet
        A pandapower network **after** a successful pp.runpp / pp.runopp / pp.runpm_ac_opf.

    Returns
    -------
    float 'penalty'
        Sum of all scaled constraint violations (>= 0). Zero implies all checked
        constraints are satisfied within `ctol`.
    """
    pen = 0.0
    baseMVA = float(net.sn_mva)  # to mimic standval scaling in DeepOPF
   
    # --------- Gen P/Q (divide by baseMVA like DeepOPF) ----------
    if len(net.gen):
        g = net.gen
        rg = net.res_gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool)

        if {"min_p_mw","max_p_mw"}.issubset(g.columns):
            p   = rg["p_mw"].to_numpy()[on]
            pmin= g["min_p_mw"].to_numpy()[on]
            pmax= g["max_p_mw"].to_numpy()[on]
            u = np.maximum(p - pmax, 0.0) / baseMVA; u[u<ctol]=0.0
            l = np.maximum(pmin - p, 0.0) / baseMVA; l[l<ctol]=0.0
            pen += u.sum() + l.sum()

        if {"min_q_mvar","max_q_mvar"}.issubset(g.columns):
            q   = rg["q_mvar"].to_numpy()[on]
            qmin= g["min_q_mvar"].to_numpy()[on]
            qmax= g["max_q_mvar"].to_numpy()[on]
            u = np.maximum(q - qmax, 0.0) / baseMVA; u[u<ctol]=0.0
            l = np.maximum(qmin - q, 0.0) / baseMVA; l[l<ctol]=0.0
            pen += u.sum() + l.sum()

    # --------- Voltage on all buses   ----------
    # Note: original paper uses only PQ buses here, we use all to 
    # do a fair evaluation with other baselines
    if len(net.bus):
        v = net.res_bus["vm_pu"].to_numpy()

        v_lo = net.bus.get("min_vm_pu", 0.95)
        v_hi = net.bus.get("max_vm_pu", 1.05)
        v_lo = (v_lo if np.isscalar(v_lo) else v_lo.to_numpy())
        v_hi = (v_hi if np.isscalar(v_hi) else v_hi.to_numpy())

        u = np.maximum(v - v_hi, 0.0); u[u<ctol]=0.0
        l = np.maximum(v_lo - v, 0.0); l[l<ctol]=0.0
        pen += u.sum() + l.sum()

    # --------- Lines: use apparent power vs an inferred Smax; penalize thermal overload----------
    # DeepOPF (Ff, Ft): max(0, |S_end| - rateA)/baseMVA at both ends
    if len(net.line):
        rl = net.res_line
        # Computes apparent power at each end of each pandapower line
        S_from = np.sqrt(rl["p_from_mw"]**2 + rl["q_from_mvar"]**2).to_numpy()
        S_to   = np.sqrt(rl["p_to_mw"]**2   + rl["q_to_mvar"]**2).to_numpy()
        lpct   = rl["loading_percent"].to_numpy()  # = 100 * S / Smax

        # sanitize base arrays
        S_from = np.nan_to_num(S_from, nan=0.0, posinf=0.0, neginf=0.0)
        S_to   = np.nan_to_num(S_to,   nan=0.0, posinf=0.0, neginf=0.0)
        lpct   = np.nan_to_num(lpct,   nan=0.0, posinf=0.0, neginf=0.0)

        # avoid lpct <= 0  
        lpct = np.maximum(lpct, 1e-3)

        # infer Smax from loading percent (Smax = S * 100 / loading%)
        # avoid div/0; if lpct==0, treat as no limit info -> zero over
        eps = 1e-12
        Smax_from = np.where(lpct>eps, S_from * 100.0 / lpct, np.inf)
        Smax_to   = np.where(lpct>eps, S_to   * 100.0 / lpct, np.inf)
        #Overload per end
        over_from = np.maximum(S_from - Smax_from, 0.0) / baseMVA
        over_to   = np.maximum(S_to   - Smax_to,   0.0) / baseMVA
        over_from[over_from<ctol]=0.0
        over_to[over_to<ctol]=0.0
        pen += over_from.sum() + over_to.sum()

    # --------- 2-winding Transformers: same idea ----------
    if len(net.trafo):
        rt = net.res_trafo
        S_hv = np.sqrt(rt["p_hv_mw"]**2 + rt["q_hv_mvar"]**2).to_numpy()
        lpct = rt["loading_percent"].to_numpy()
        eps = 1e-12
        Smax = np.where(lpct>eps, S_hv * 100.0 / lpct, np.inf)
        over = np.maximum(S_hv - Smax, 0.0) / baseMVA
        over[over<ctol]=0.0
        pen += over.sum()

    # --------- 3-winding Transformers (if present) ----------
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        rt3 = net.res_trafo3w
        # use hv side as proxy; pp’s loading_percent is already max over legs,
        # but we reconstruct S like above for matpower-equivalent scaling
        S_hv = np.sqrt(rt3["p_h_mw"]**2 + rt3["q_h_mvar"]**2).to_numpy()
        lpct = rt3["loading_percent"].to_numpy()
        eps = 1e-12
        Smax = np.where(lpct>eps, S_hv * 100.0 / lpct, np.inf)
        over = np.maximum(S_hv - Smax, 0.0) / baseMVA
        over[over<ctol]=0.0
        pen += over.sum()

    return float(pen)



# -----------------------------
# Custom autograd layer (zero-order gradient)
# -----------------------------
class PenaltyACPF(Function):
    @staticmethod
    def forward(ctx, y_norm, x_norm, ctx_dict):
        """
        Returns scalar penalty (sum over batch) to be added to loss.
        """
        ctx.save_for_backward(y_norm, x_norm)
        ctx.ctx_dict = ctx_dict

        device = y_norm.device
        B = y_norm.shape[0]
        penalties = []

        for i in range(B):
            pen_i = PenaltyACPF._penalty_single_sample(y_norm[i], x_norm[i], ctx_dict)
            penalties.append(pen_i)

        total_pen = torch.stack(penalties).sum().to(dtype=torch.float32, device=device)
        return total_pen

    @staticmethod
    def backward(ctx, grad_output):
        y_norm, x_norm = ctx.saved_tensors
        C = ctx.ctx_dict
        device = y_norm.device
        B, Ny = y_norm.shape

        # Random direction per sample
        v = torch.randn_like(y_norm)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        h = 1e-4

        y_plus  = y_norm + h * v
        y_minus = y_norm - h * v

        p_plus, p_minus = [], []
        for i in range(B):
            p_plus.append(PenaltyACPF._penalty_single_sample(y_plus[i],  x_norm[i], C))
            p_minus.append(PenaltyACPF._penalty_single_sample(y_minus[i], x_norm[i], C))

        p_plus  = torch.stack(p_plus).to(device=device, dtype=torch.float32)
        p_minus = torch.stack(p_minus).to(device=device, dtype=torch.float32)

        # Directional derivative per-sample
        dir_deriv = (p_plus - p_minus) / (2.0 * h)  # (B,)

        S = len(C["active_gen_idx"]) + len(C["pv_bus_list"]) + 1  # ~= (#active gens + #PV buses + slack)
        go = float(grad_output)  # scalar since forward returns sum
        grad_y = go * (S * dir_deriv).view(-1,1) * v   
        # No gradient wrt inputs or ctx
        grad_x = torch.zeros_like(x_norm)
        return grad_y, grad_x, None

    @staticmethod
    def _penalty_single_sample(y_norm_i, x_norm_i, C):
        # Unpack context
        base_net: pp.pandapowerNet = C["base_net"]
        P_mu = C["P_mu"]; P_sig = C["P_sig"]
        Q_mu = C["Q_mu"]; Q_sig = C["Q_sig"]
        active_gen_idx = C["active_gen_idx"]
        pv_bus_list = C["pv_bus_list"]
        slack_bus = C["slack_bus"]
        PG_lo = C["PG_lo"]; PG_hi = C["PG_hi"]
        Slack_lo = C["Slack_lo"]; Slack_hi = C["Slack_hi"]
        Vm_lo = C["Vm_lo"]; Vm_hi = C["Vm_hi"]

        # Split y (normalized [0,1])
        n_pg = len(active_gen_idx)
        n_pv = len(pv_bus_list)
        y = y_norm_i.detach().cpu().numpy()
        PG_norm   = y[0:n_pg]
        Slack_norm = y[n_pg]
        Vm_norm   = y[n_pg+1 : n_pg+1+n_pv]

        # Denormalize outputs
        PG = PG_lo + PG_norm * (PG_hi - PG_lo)                # MW
        SlackV = Slack_lo + Slack_norm * (Slack_hi - Slack_lo) # pu
        Vm_PV = Vm_lo + Vm_norm * (Vm_hi - Vm_lo)             # pu

        # Denormalize inputs (loads)
        x = x_norm_i.detach().cpu().numpy()
        n_bus = len(base_net.bus)
        Pz = x[0:n_bus]
        Qz = x[n_bus:2*n_bus]
        P_D = P_mu + Pz * P_sig
        Q_D = Q_mu + Qz * Q_sig

        # Build case
        net = copy.deepcopy(base_net)
        load_map = C["load_bus_to_idxs"]
        pv_pos   = C["pv_pos"]

        # Set loads per bus (even split across loads on same bus)
        if len(net.load):
            for bus in net.bus.index:
                idxs = load_map.get(int(bus))
                if idxs is None or len(idxs) == 0: 
                    continue
                share_p = float(P_D[bus]) / len(idxs)
                share_q = float(Q_D[bus]) / len(idxs)
                net.load.loc[idxs, "p_mw"]   = share_p
                net.load.loc[idxs, "q_mvar"] = share_q

        # Set generator P (active gens only)
        for k, gi in enumerate(active_gen_idx):
            net.gen.at[gi, "p_mw"] = float(PG[k])

        # Slack voltage target (ext_grid at slack bus)
        if len(net.ext_grid):
            eg_idx = net.ext_grid.index[net.ext_grid["bus"] == slack_bus]
            if len(eg_idx):
                net.ext_grid.at[eg_idx[0], "vm_pu"] = float(SlackV)

        # PV bus voltage targets (gens define bus setpoints)
        for k, gi in enumerate(active_gen_idx):
            bus_k = int(net.gen.at[gi, "bus"])
            pos = pv_pos.get(bus_k)
            if pos is not None:
                net.gen.at[gi, "vm_pu"] = float(Vm_PV[pos])

         
        PEN_FAIL = 1.0
        prep_constraints(net)
        try:
            run_pf(net, vm_init=C.get("vm_init"), va_init=C.get("va_init"))
        except Exception:
            return torch.tensor(PEN_FAIL, dtype=torch.float32)

        if not getattr(net, "converged", True):
            return torch.tensor(PEN_FAIL, dtype=torch.float32)

        pen = compute_penalty_from_net(net)

        if not np.isfinite(pen):
            pen = PEN_FAIL

        return torch.tensor(pen, dtype=torch.float32)

# -----------------------------
# NN: maps normalized loads -> normalized [PG_active, SlackV, Vm_PV]
# -----------------------------
class DeepOPFPenaltyNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 2*hidden)
        self.l3 = nn.Linear(2*hidden, hidden)
        self.l4 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l3(h))
        y = torch.sigmoid(self.l4(h))  # normalized outputs in [0,1]
        return y

# -----------------------------
# Training wrapper
# -----------------------------
def train_penalty_model(base_net, X_t, Y_t, meta, epochs=20, batch_size=3072, lr=1e-3, device="cuda", w1= 1, w2=0.1):
    """
    X_t: tensor (N, 2*n_bus)  -> inputs [Pz, Qz]
    Y_t: tensor (N, Ny)       -> labels normalized like the outputs [PG_active_norm, SlackV_norm, Vm_PV_norm]
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    N, in_dim = X_t.shape

    n_pg = len(meta["active_gen_idx"])
    n_pv = len(meta["pv_bus_list"])
    out_dim = n_pg + 1 + n_pv

    model = DeepOPFPenaltyNet(in_dim, hidden=vars.model_hidden, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss(reduction="mean")

    #for faster computation, precompute the 2 maps
    load_bus_to_idxs = {}
    if len(base_net.load):
        buses = base_net.load["bus"].to_numpy()
        idxs  = base_net.load.index.to_numpy()
        for b in base_net.bus.index.to_numpy():
            m = (buses == b)
            if m.any():
                load_bus_to_idxs[int(b)] = idxs[m]

    pv_pos = {int(b): i for i, b in enumerate(meta["pv_bus_list"])}

    # Historical-average init (to initalize Newton-Raphson's method inside runpp
    # with historical avg. values of dependent variables form training data)
    vm_init, va_init = build_hist_initialization(base_net, meta, Y_t)
    # Pack context for penalty layer
    ctx_dict = {
        "base_net": base_net,
        "P_mu": meta["P_mu"], "P_sig": meta["P_sig"],
        "Q_mu": meta["Q_mu"], "Q_sig": meta["Q_sig"],
        "active_gen_idx": meta["active_gen_idx"],
        "pv_bus_list": meta["pv_bus_list"],
        "slack_bus": meta["slack_bus"],
        "PG_lo": meta["PG_lo"], "PG_hi": meta["PG_hi"],
        "Slack_lo": meta["Slack_lo"], "Slack_hi": meta["Slack_hi"],
        "Vm_lo": meta["Vm_lo"], "Vm_hi": meta["Vm_hi"],
        "load_bus_to_idxs": load_bus_to_idxs,
        "pv_pos": pv_pos,
        "vm_init": vm_init,
        "va_init": va_init
    }

    dl = DataLoader(TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_mse = 0.0
        total_pen = 0.0
        step=0
        for xb, yb in dl:
            step+=1
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            y_pred = model(xb)                      # normalized preds
            pen = PenaltyACPF.apply(y_pred, xb, ctx_dict)  # scalar
            mse = criterion(y_pred, yb)
            loss = w1*mse + w2*pen
            # ---- PRINT HERE ----
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

        print(f"[epoch {ep+1:03d}] mse = {total_mse/len(dl):.6f} ; penalty = {total_pen/len(dl):.6f}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--gen_data", action='store_true', default=False)
    args = parser.parse_args()

    sample_net, _ = util.get_base_net(args.case, 'in_distribution')
    bs = 3072

    fname= "src/baselines/DeepOPF/datasets/" + args.case + "/training_dataset"
    if args.gen_data:
        loader, meta = build_deepopf_training_data_loader(sample_net, args.case, fname, batch_size=bs)
    else:
        loaded = torch.load(fname+".pt", map_location="cpu")
        dataset = loaded["dataset"]          # should be (X, Y)
        meta = loaded["meta"]
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    # Collect X, Y from loader (use labels for MSE term)
    X_list, Y_list = [], []
    for xb, yb in loader:
        X_list.append(xb)
        Y_list.append(yb)
    X_t = torch.cat(X_list, dim=0).float()
    Y_t = torch.cat(Y_list, dim=0).float()

    model = train_penalty_model(
        base_net=sample_net,
        X_t=X_t,
        Y_t=Y_t,
        meta=meta,
        epochs=10,
        batch_size=3072,
        lr=1e-3,
        device="cuda",
        w1= 1,
        w2= 0.1,   
    )

    # Save weights  
    torch.save(model.state_dict(), f"src/baselines/DeepOPF/model/{args.case}/trained_model.pth")
