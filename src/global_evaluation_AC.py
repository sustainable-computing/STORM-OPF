# global_evaluation.py: contains all necessary functions for model evaluation
#for AC-OPF
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple
import src.global_vars as gvars
import src.util as util
import numpy as np
import pandapower as pp
import pandas as pd
try:
    import torch
    from torch import nn
except Exception:  # allow non-torch use
    torch = None
    nn = None


# -----------------------------
# Penalty & Cost (PF-dependent)
# -----------------------------
def compute_penalty_from_net(net: pp.pandapowerNet, ctol= gvars.ctol) -> float:
    """
    Aggregate a scalar nonnegative penalty from constraint violations on a solved net:
      - Gen P/Q bounds
      - PQ-bus voltage bounds
      - Line/trafo loading (apparent power vs inferred Smax from loading%)
    Assumes pandapower results tables are populated (after pp.runpp/pp.runopp).
    """
    pen = 0.0
    baseMVA = float(net.sn_mva)

    # ---- Gen P/Q ----
    if len(net.gen):
        g = net.gen
        rg = net.res_gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool)

        if {"min_p_mw", "max_p_mw"}.issubset(g.columns):
            p = rg["p_mw"].to_numpy()[on]
            pmin = g["min_p_mw"].to_numpy()[on]
            pmax = g["max_p_mw"].to_numpy()[on]
            u = np.maximum(p - pmax, 0.0) / baseMVA; u[u < ctol] = 0.0
            l = np.maximum(pmin - p, 0.0) / baseMVA; l[l < ctol] = 0.0
            pen += u.sum() + l.sum()

        if {"min_q_mvar", "max_q_mvar"}.issubset(g.columns):
            q = rg["q_mvar"].to_numpy()[on]
            qmin = g["min_q_mvar"].to_numpy()[on]
            qmax = g["max_q_mvar"].to_numpy()[on]
            u = np.maximum(q - qmax, 0.0) / baseMVA; u[u < ctol] = 0.0
            l = np.maximum(qmin - q, 0.0) / baseMVA; l[l < ctol] = 0.0
            pen += u.sum() + l.sum()

    # ---- PQ-bus voltages ----
    if len(net.bus):
        vm = net.res_bus["vm_pu"].to_numpy()

        has_ext = np.zeros(len(net.bus), dtype=bool)
        if len(net.ext_grid): has_ext[net.ext_grid["bus"].to_numpy()] = True

        has_gen = np.zeros(len(net.bus), dtype=bool)
        if len(net.gen):
            gen_buses = net.gen.loc[net.gen["in_service"].fillna(True), "bus"].to_numpy()
            has_gen[gen_buses] = True

        is_pq = ~(has_ext | has_gen)

        v_lo = net.bus.get("min_vm_pu", 0.95)
        v_hi = net.bus.get("max_vm_pu", 1.05)
        v_lo = (v_lo if np.isscalar(v_lo) else v_lo.to_numpy())[is_pq]
        v_hi = (v_hi if np.isscalar(v_hi) else v_hi.to_numpy())[is_pq]
        v = vm[is_pq]

        u = np.maximum(v - v_hi, 0.0); u[u < ctol] = 0.0
        l = np.maximum(v_lo - v, 0.0); l[l < ctol] = 0.0
        pen += u.sum() + l.sum()

    # ---- Lines ----
    if len(net.line):
        rl = net.res_line
        S_from = np.sqrt(rl["p_from_mw"]**2 + rl["q_from_mvar"]**2).to_numpy()
        S_to = np.sqrt(rl["p_to_mw"]**2   + rl["q_to_mvar"]**2).to_numpy()
        lpct = rl["loading_percent"].to_numpy()
        eps = 1e-12
        Smax_from = np.where(lpct > eps, S_from * 100.0 / lpct, np.inf)
        Smax_to = np.where(lpct > eps, S_to   * 100.0 / lpct, np.inf)
        over_from = np.maximum(S_from - Smax_from, 0.0) / baseMVA
        over_to = np.maximum(S_to   - Smax_to,   0.0) / baseMVA
        over_from[over_from < ctol] = 0.0
        over_to[over_to < ctol] = 0.0
        pen += over_from.sum() + over_to.sum()

    # ---- 2-winding trafos ----
    if len(net.trafo):
        rt = net.res_trafo
        S_hv = np.sqrt(rt["p_hv_mw"]**2 + rt["q_hv_mvar"]**2).to_numpy()
        lpct = rt["loading_percent"].to_numpy()
        eps = 1e-12
        Smax = np.where(lpct > eps, S_hv * 100.0 / lpct, np.inf)
        over = np.maximum(S_hv - Smax, 0.0) / baseMVA
        over[over < ctol] = 0.0
        pen += over.sum()

    # ---- 3-winding trafos ----
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        rt3 = net.res_trafo3w
        S_hv = np.sqrt(rt3["p_h_mw"]**2 + rt3["q_h_mvar"]**2).to_numpy()
        lpct = rt3["loading_percent"].to_numpy()
        eps = 1e-12
        Smax = np.where(lpct > eps, S_hv * 100.0 / lpct, np.inf)
        over = np.maximum(S_hv - Smax, 0.0) / baseMVA
        over[over < ctol] = 0.0
        pen += over.sum()

    return float(pen)


def evaluate_total_cost(net: pp.pandapowerNet) -> float:
    """
    Return the total quadratic cost using net.poly_cost and solved results.
    Supports 'gen' and 'ext_grid' entries.
    """
    total = 0.0
    if not hasattr(net, "poly_cost") or not len(net.poly_cost):
        return float("nan")

    for _, row in net.poly_cost.iterrows():
        et = row.get("et")  # "gen" or "ext_grid"
        eid = int(row.get("element"))
        if et == "gen":
            p = float(net.res_gen.at[eid, "p_mw"])
        elif et == "ext_grid":
            p = float(net.res_ext_grid.at[eid, "p_mw"])
        else:
            continue

        cp0 = float(row.get("cp0_eur", 0.0))
        cp1 = float(row.get("cp1_eur_per_mw", 0.0))
        cp2 = float(row.get("cp2_eur_per_mw2", 0.0))
        total += cp0 + cp1 * p + cp2 * (p ** 2)
    return float(total)



def get_orig_pq_bus_idx(net):
    """
    'Load/PQ buses' in the sense of: buses that are NOT ext_grid buses and NOT gen buses
    
    Returns actual pandapower bus indices (not positional 0..N-1).
    """
    bus_idx = net.bus.index
    bus_pos = pd.Index(bus_idx)

    has_ext = np.zeros(len(bus_idx), dtype=bool)
    if len(net.ext_grid):
        eg = net.ext_grid
        on = eg["in_service"].fillna(True).to_numpy().astype(bool) if "in_service" in eg.columns else np.ones(len(eg), bool)
        ext_b = eg.loc[on, "bus"].to_numpy()
        pos = bus_pos.get_indexer(ext_b)
        pos = pos[pos >= 0]
        has_ext[pos] = True

    has_gen = np.zeros(len(bus_idx), dtype=bool)
    if len(net.gen):
        g = net.gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool) if "in_service" in g.columns else np.ones(len(g), bool)
        gen_b = g.loc[on, "bus"].to_numpy()
        pos = bus_pos.get_indexer(gen_b)
        pos = pos[pos >= 0]
        has_gen[pos] = True

    is_pq = ~(has_ext | has_gen)
    return bus_idx[is_pq].to_numpy()


def is_feasible_by_hard_constraints(
    net: pp.pandapowerNet,
    vm_tol: float = 1e-3,     # per-unit slack for voltage limits
    pct_tol: float = 1e-3,    # percent slack for loading (e.g., allow 100.001%)
    pq_tol: float = 1e-6,      # MW/Mvar absolute floor for generator P/Q bounds
    require_converged: bool = True,
    vm_bus_idx= None
) -> bool:
    """
    Return True iff all inequality constraints are satisfied within the given numeric tolerances.
    Assumes pp.runpp() has been called and results tables are populated.
    """
    # PF must have converged
    if require_converged: 
        pf_conv = bool(getattr(net, "converged", False)) 
        opf_conv = bool(net.get("OPF_converged", False)) 
        #print("pf_conv:", pf_conv, "| opf_conv:", opf_conv) 
        # # Only fail if neither PF nor OPF is converged 
        if not (pf_conv or opf_conv): 
            print("ret false line 157: PF/OPF not converged") 
            return False

    # ---------------- Bus voltage bounds ----------------
    if len(net.bus):
        bus_idx = net.bus.index

        # ensure ordering matches net.bus
        vm = net.res_bus.loc[bus_idx, "vm_pu"].to_numpy()

        v_lo_col = net.bus.get("min_vm_pu", 0.95)
        v_hi_col = net.bus.get("max_vm_pu", 1.05)
        v_lo = (v_lo_col if np.isscalar(v_lo_col) else v_lo_col.to_numpy())
        v_hi = (v_hi_col if np.isscalar(v_hi_col) else v_hi_col.to_numpy())

        # select subset
        if vm_bus_idx is not None:
            mask = bus_idx.isin(np.asarray(vm_bus_idx))
            vm = vm[mask]
            v_lo = v_lo[mask]
            v_hi = v_hi[mask]

        viol_vm_lo = (vm + vm_tol) < v_lo
        viol_vm_hi = (vm - vm_tol) > v_hi
        if np.any(viol_vm_lo) or np.any(viol_vm_hi):
            #print("VIOLATION: voltage (selected buses)")
            return False


    # ---------------- Generator P/Q limits (relative-or-absolute tolerances) ----------------
    # Use pq_tol as absolute floor; add relative slack of 0.1% per element.
    pq_rel = 1e-3  # 0.1% relative tolerance
    if len(net.gen):
        g = net.gen
        rg = net.res_gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool)

        # Active power
        if {"min_p_mw", "max_p_mw"}.issubset(g.columns):
            p = rg.loc[on, "p_mw"].to_numpy()
            pmin = g.loc[on, "min_p_mw"].to_numpy()
            pmax = g.loc[on, "max_p_mw"].to_numpy()
            has_pmin = ~np.isnan(pmin)
            has_pmax = ~np.isnan(pmax)

            # p > pmax
            if np.any(has_pmax):
                p_u = p[has_pmax]
                pmax_u = pmax[has_pmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(pmax_u), 1.0), pq_tol)
                if np.any((p_u - pmax_u) > tol_u):
                    #print('VIOLATION l205: p>pmax')
                    return False

            # p < pmin
            if np.any(has_pmin):
                p_l = p[has_pmin]
                pmin_l = pmin[has_pmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(pmin_l), 1.0), pq_tol)
                if np.any((pmin_l - p_l) > tol_l):
                    #print('VIOLATION l214: p<pmin')
                    return False

        # Reactive power
        if {"min_q_mvar", "max_q_mvar"}.issubset(g.columns):
            q = rg.loc[on, "q_mvar"].to_numpy()
            qmin = g.loc[on, "min_q_mvar"].to_numpy()
            qmax = g.loc[on, "max_q_mvar"].to_numpy()
            has_qmin = ~np.isnan(qmin)
            has_qmax = ~np.isnan(qmax)

            # q > qmax
            if np.any(has_qmax):
                q_u = q[has_qmax]
                qmax_u = qmax[has_qmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(qmax_u), 1.0), pq_tol)
                if np.any((q_u - qmax_u) > tol_u):
                    #print('VIOLATION l231: q>qmax')
                    return False

            # q < qmin
            if np.any(has_qmin):
                q_l = q[has_qmin]
                qmin_l = qmin[has_qmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(qmin_l), 1.0), pq_tol)
                if np.any((qmin_l - q_l) > tol_l):
                    #print('VIOLATION l240: q<qmin')
                    return False

    # ---------------- Static generator (sgen) P/Q limits ----------------
    if len(net.sgen) and hasattr(net, "res_sgen") and len(net.res_sgen):
        s = net.sgen
        rs = net.res_sgen
        on = s["in_service"].fillna(True).to_numpy().astype(bool)

        # Active power
        if {"min_p_mw", "max_p_mw"}.issubset(s.columns):
            p = rs.loc[on, "p_mw"].to_numpy()
            pmin = s.loc[on, "min_p_mw"].to_numpy()
            pmax = s.loc[on, "max_p_mw"].to_numpy()
            has_pmin = ~np.isnan(pmin)
            has_pmax = ~np.isnan(pmax)

            # p > pmax
            if np.any(has_pmax):
                p_u = p[has_pmax]
                pmax_u = pmax[has_pmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(pmax_u), 1.0), pq_tol)
                if np.any((p_u - pmax_u) > tol_u):
                    #print('VIOLATION l263: sgen p>pmax')
                    return False

            # p < pmin
            if np.any(has_pmin):
                p_l = p[has_pmin]
                pmin_l = pmin[has_pmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(pmin_l), 1.0), pq_tol)
                if np.any((pmin_l - p_l) > tol_l):
                    #print('VIOLATION l272: sgen p<pmin')
                    return False

        # Reactive power
        if {"min_q_mvar", "max_q_mvar"}.issubset(s.columns):
            q = rs.loc[on, "q_mvar"].to_numpy()
            qmin = s.loc[on, "min_q_mvar"].to_numpy()
            qmax = s.loc[on, "max_q_mvar"].to_numpy()
            has_qmin = ~np.isnan(qmin)
            has_qmax = ~np.isnan(qmax)

            # q > qmax
            if np.any(has_qmax):
                q_u = q[has_qmax]
                qmax_u = qmax[has_qmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(qmax_u), 1.0), pq_tol)
                if np.any((q_u - qmax_u) > tol_u):
                    #print('VIOLATION l289: q>qmax')
                    return False

            # q < qmin
            if np.any(has_qmin):
                q_l = q[has_qmin]
                qmin_l = qmin[has_qmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(qmin_l), 1.0), pq_tol)
                if np.any((qmin_l - q_l) > tol_l):
                    #print('VIOLATION l298: q<qmin')
                    return False

    def _violates_loading(res_tbl, elem_tbl, on_mask, pct_tol):
        # prefer per-element max_loading_percent; default 100 if missing/NaN
        limit = (elem_tbl.get("max_loading_percent", 100.0)
                .fillna(100.0).to_numpy())[on_mask]
        lpct = res_tbl.loc[on_mask, "loading_percent"].to_numpy()
        ok_mask = ~np.isnan(lpct) & ~np.isnan(limit)
        return np.any(lpct[ok_mask] > (limit[ok_mask] + pct_tol))

    # Line thermal loading: loading_percent <= max_loading or 100%
    if len(net.line) and hasattr(net, "res_line") and len(net.res_line):
        on = net.line["in_service"].fillna(True).to_numpy().astype(bool)
        if _violates_loading(net.res_line, net.line, on, pct_tol):
            #print('VIOLATION l313: lpct')
            return False

    # 2-w trafos loading limit
    if len(net.trafo) and hasattr(net, "res_trafo") and len(net.res_trafo):
        on = net.trafo["in_service"].fillna(True).to_numpy().astype(bool)
        if _violates_loading(net.res_trafo, net.trafo, on, pct_tol):
            #print('VIOLATION l320: 2w trafo')
            return False

    # 3-w trafos loading limit
    if hasattr(net, "trafo3w") and len(net.trafo3w) and hasattr(net, "res_trafo3w") and len(net.res_trafo3w):
        on = net.trafo3w["in_service"].fillna(True).to_numpy().astype(bool)
        if _violates_loading(net.res_trafo3w, net.trafo3w, on, pct_tol):
            #print('VIOLATION l327: 3wtrafo')
            return False
    #print('l 318: ret true')
    return True


# --------------------------
# IPOPT Baseline evaluator
#---------------------------
def evaluate_opf_baseline(
    net_paths: List[str],
    run_opf_fn: Callable[[pp.pandapowerNet], None],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate an exact AC-OPF baseline (e.g., IPOPT) on a set of scenario nets.

    Args:
        net_paths: list of paths to pandapowerNet pickle files / scenario files.
        run_opf_fn: function that runs OPF on a net (default: run_opf above).
        feas_tol: penalty threshold used for 'soft' feasibility (same as PFEvaluator).
        max_penalty: penalty to assign when OPF fails.
        ctol: tolerance passed to compute_penalty_from_net.
        verbose: whether to print warnings.

    Returns:
        {
          'n_samples',
          'costs',
          'hard_feasible_flags',     # strict constraint check
          'feasibility_rate_%',
        }
    """
    n_samples = 0
    costs: List[float] = []
    hard_feasible_flags: List[bool] = []

    opf_time_sum = 0.0
    opf_time_kept = 0

    for idx, path in enumerate(net_paths):
        n_samples += 1
        if verbose:
            print(f"[OPF] ({idx+1}/{len(net_paths)}) scenario: {path}")

        # Load base net
        try:
            net = util.load_net(path)
            # store before running OPF (so later checks are stable)
            net["_orig_pq_bus_idx"] = get_orig_pq_bus_idx(net)
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load net from {path}: {e}")
        
            costs.append(float("nan"))
            hard_feasible_flags.append(False)
            continue

        # Run OPF
        c_val = float("nan")
        hard_feas = False
        try:
            t0 = time.perf_counter()
            run_opf_fn(net)  #  iterative OPF
            t1 = time.perf_counter()
            solve_time = t1 - t0

            # Cost & penalty
            converged = bool(getattr(net, "converged", False)) or bool(net.get("OPF_converged", False))
            if converged:
                opf_time_sum += solve_time
                opf_time_kept += 1
                c_val = evaluate_total_cost(net)
            else:
                c_val = float("nan")
            hard_feas = is_feasible_by_hard_constraints(net, vm_bus_idx=net["_orig_pq_bus_idx"])
        except Exception as e:
            if verbose:
                print(f"[WARN] OPF / metrics failed on {path}: {e}")

        costs.append(c_val)
        hard_feasible_flags.append(hard_feas)

    def _rate(flags: List[bool]) -> float:
        return 100.0 * float(np.mean(flags)) if flags else float("nan")

    avg_cost = float(np.mean([c for c in costs if np.isfinite(c)])) if len(costs) else float("nan")
    avg_opf_solve_time = (opf_time_sum / opf_time_kept) if opf_time_kept > 0 else float("nan")
    
    results = {
        "n_samples": n_samples,
        "costs": costs,
        "avg_cost": avg_cost,
        "hard_feasible_flags": hard_feasible_flags,
        "feasibility_rate_%": _rate(hard_feasible_flags),
        "infer_time_per_sample_s": avg_opf_solve_time
    }
    print(results['feasibility_rate_%'])
    return results


# -----------------------------
# Generic PF-based Evaluator
# -----------------------------
class PFEvaluator:
    """
    Adaptable evaluator for AC-OPF surrogates/algorithms.

    To evaluate: must provide two algorithm-specific callbacks:
      - denorm_fn(y_norm, meta, device) -> Any
          (e.g., return (PG, SlackV, Vm_PV) tensors for DeepOPF)
      - apply_pred_fn(base_net, meta, x_i, y_pred_i, denorm_fn, device) -> pandapowerNet
          Builds a **fresh** net with the ith prediction applied. Should call prep constraints if needed.

    Optionally, provide model_infer_fn if your model doesn't support plain `model(xb)`.
    """

    def __init__(
        self,
        denorm_fn: Callable[[Any, Dict[str, Any], str], Any],
        apply_pred_fn: Callable[[pp.pandapowerNet, Dict[str, Any], Any, Any, Callable, str], pp.pandapowerNet],
        model_infer_fn: Callable[[Any, Any, str], Any] | None = None,
        feas_tol= gvars.feas_tol,
        max_penalty= gvars.max_pen,
        ctol = gvars.ctol,
        runpp_kwargs: Dict[str, Any] | None = None,
    ):
        self.denorm_fn = denorm_fn
        self.apply_pred_fn = apply_pred_fn
        self.model_infer_fn = model_infer_fn
        self.feas_tol = float(feas_tol)  #required for soft penalty calculation (following DeepOPF), not required for hard feasibility implemented here
        self.max_penalty = float(max_penalty)
        self.ctol = float(ctol)
        self.runpp_kwargs = dict(enforce_q_lims=True, calculate_voltage_angles=True)
        if runpp_kwargs:
            self.runpp_kwargs.update(runpp_kwargs)

    # --- internal helpers ---
    def _is_torch(self, obj) -> bool:
        return (torch is not None) and hasattr(obj, "__class__") and "torch" in str(obj.__class__.__module__)

    def _default_model_infer(self, model: Any, xb: Any, device: str):
        if self._is_torch(model):
            xb_t = xb.to(device) if hasattr(xb, "to") else xb
            model_device = next(model.parameters()).device if hasattr(model, "parameters") else device
            if device != str(model_device):
                pass
            return model(xb_t)
        else:
            # numpy/other model; assume callable
            return model(xb)

    def _run_pf(self, net: pp.pandapowerNet):
        pp.runpp(net, **self.runpp_kwargs)

    # --- public ---
    def evaluate(
        self,
        model: Any,
        loader: Iterable[Tuple[Any, Any]],
        meta: Dict[str, Any],
        net_paths: str,
        device: str = "cpu",
        collect_norm_mse: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        
        
        n_samples = 0
        costs: List[float] = []
        hard_feasible_flags: List[bool] = []
        mse_norm_sum = 0.0
        use_torch = (torch is not None) and self._is_torch(model)
        mse_loss = nn.MSELoss(reduction="mean") if (collect_norm_mse and use_torch) else None

        infer_time_sum = 0.0
        infer_kept_count = 0
        
        net_path_ind= 0
        for xb, yb in loader:
            # Inference (batched)
            t0 = time.perf_counter()
            y_pred = (
                self.model_infer_fn(model, xb, device)
                if self.model_infer_fn is not None
                else self._default_model_infer(model, xb, device)
            )
            #print(y_pred)
            
            t1 = time.perf_counter()
            batch_infer_time = (t1 - t0)   
            # Norm-space MSE (optional)
            if collect_norm_mse and use_torch:
                yb_t = yb.to(device).float()
                y_pred_t = y_pred
                mse_norm_sum += float(mse_loss(y_pred_t, yb_t).item()) * yb_t.size(0)

            # Per-sample PF & metrics
            B = y_pred.shape[0] if hasattr(y_pred, "shape") else len(y_pred)
            per_sample_infer = batch_infer_time / max(1, B)

            for i in range(B): #for each sample in this batch
                n_samples += 1

                #load the corresponding net
                loaded_net= util.load_net(net_paths[net_path_ind])
                net_path_ind += 1
                # Slice ith sample
                x_i = xb[i] if hasattr(xb, "__getitem__") else xb
                y_i = y_pred[i] if hasattr(y_pred, "__getitem__") else y_pred

                # Build fresh net with prediction applied
                try:
                    net_i = self.apply_pred_fn(copy.deepcopy(loaded_net), meta, x_i, y_i, self.denorm_fn, device)
                    if "_orig_pq_bus_idx" not in net_i:
                        net_i["_orig_pq_bus_idx"] = get_orig_pq_bus_idx(net_i)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] apply_pred_fn failed on sample {i}: {e}")
                    
                    costs.append(float("nan"))
                    hard_feasible_flags.append(False)
                    continue

                # Solve PF and compute penalty/cost
                c_val = float("nan")
                try:
                    self._run_pf(net_i)     #do powerflow by calling pp.runpp
                    c_val = evaluate_total_cost(net_i)
            
                except Exception as e:
                    if verbose:
                        print(f"[WARN] runpp/metrics failed on sample {i}: {e}")

                costs.append(c_val)
                
                isFeas= is_feasible_by_hard_constraints(net_i,vm_bus_idx=net_i["_orig_pq_bus_idx"])
                hard_feasible_flags.append(isFeas)
                #print(isFeas)
                infer_time_sum += per_sample_infer
                infer_kept_count += 1
                
        avg_cost = float(np.nanmean(costs)) if len(costs) else float("nan")
        infer_time_per_sample = (infer_time_sum / infer_kept_count) if infer_kept_count > 0 else float("nan")

        results = {
            "n_samples": n_samples,
            "costs": costs,
            "avg_cost": avg_cost, #don't consider feasibility here
            "hard_feasible_flags": hard_feasible_flags,
            "feasibility_rate_%": 100.0 * float(np.mean(hard_feasible_flags)) if hard_feasible_flags else float("nan"),
            "infer_time_per_sample_s": infer_time_per_sample,
            "infer_time_kept_n": infer_kept_count
        }
        if collect_norm_mse and n_samples > 0:
            results["mse_norm"] = mse_norm_sum / n_samples
        
        return results
