# global_evaluation_ACSC.py: contains all necessary functions for model evaluation
#for SC AC-OPF
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandapower as pp

import src.global_vars as gvars
import src.util as util
from src.ac_utils import (
    apply_contingency,
    compute_penalty_from_net,
    compute_operating_cost,
    filter_valid_contingencies,
    get_orig_pq_bus_idx,
    is_feasible_by_hard_constraints,
    prep_constraints,
    resolve_contingencies,
    run_pf,
)

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


def evaluate_total_cost(net: pp.pandapowerNet) -> float:
    return compute_operating_cost(net)


def _contingency_eval(net_base: pp.pandapowerNet, contingencies, vm_bus_idx, ctol: float, runpp_kwargs: Dict[str, Any]):
    base_pen, _ = compute_penalty_from_net(net_base, ctol=ctol)
    base_feas = is_feasible_by_hard_constraints(net_base, vm_bus_idx=vm_bus_idx)
    vm_init = net_base.res_bus.vm_pu.values.astype(float) if hasattr(net_base, "res_bus") and len(net_base.res_bus) else None
    va_init = net_base.res_bus.va_degree.values.astype(float) if hasattr(net_base, "res_bus") and len(net_base.res_bus) and "va_degree" in net_base.res_bus else None

    max_cont_pen = 0.0
    sc_pen = float(base_pen)
    failed_count = 0
    sc_feas = bool(base_feas)
    per_cont = []

    for cont in contingencies:
        netc = copy.deepcopy(net_base)
        apply_contingency(netc, cont)
        try:
            run_pf(netc, vm_init=vm_init, va_init=va_init, enforce_q_lims=runpp_kwargs.get("enforce_q_lims", False))
            pen_c, _ = compute_penalty_from_net(netc, ctol=ctol)
            feas_c = is_feasible_by_hard_constraints(netc, vm_bus_idx=vm_bus_idx)
        except Exception:
            pen_c = float(gvars.max_pen)
            feas_c = False
        max_cont_pen = max(max_cont_pen, float(pen_c))
        sc_pen += float(pen_c)
        if not feas_c:
            failed_count += 1
            sc_feas = False
        per_cont.append({"contingency": cont, "feasible": bool(feas_c), "penalty": float(pen_c)})

    return {
        "base_penalty": float(base_pen),
        "base_feasible": bool(base_feas),
        "sc_penalty": float(sc_pen),
        "max_contingency_penalty": float(max_cont_pen),
        "failed_contingency_count": int(failed_count),
        "sc_feasible": bool(sc_feas),
        "contingency_details": per_cont,
    }


def evaluate_opf_baseline_sc(
    net_paths: List[str],
    run_opf_fn: Callable[[pp.pandapowerNet], None],
    contingency_spec=None,
    ctol: float = gvars.ctol,
    verbose: bool = True,
) -> Dict[str, Any]:
    #eval the iterative baseline 
    
    n_samples = 0
    costs: List[float] = []
    base_hard_feasible_flags: List[bool] = []
    hard_feasible_flags: List[bool] = []
    base_penalties: List[float] = []
    max_contingency_penalties: List[float] = []
    sc_penalties: List[float] = []
    failed_contingency_counts: List[int] = []
    opf_time_sum = 0.0
    opf_time_kept = 0

    for idx, path in enumerate(net_paths):
        n_samples += 1
        try:
            net = util.load_net(path)
            prep_constraints(net)
            vm_bus_idx = get_orig_pq_bus_idx(net)
            contingencies = filter_valid_contingencies(net, resolve_contingencies(net, contingency_spec=contingency_spec, scenario_path=path))
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load net from {path}: {e}")
            costs.append(float("nan"))
            base_hard_feasible_flags.append(False)
            hard_feasible_flags.append(False)
            base_penalties.append(float(gvars.max_pen))
            max_contingency_penalties.append(float(gvars.max_pen))
            sc_penalties.append(float(gvars.max_pen))
            failed_contingency_counts.append(0)
            continue

        try:
            t0 = time.perf_counter()
            run_opf_fn(net)
            t1 = time.perf_counter()
            if bool(getattr(net, "converged", False)) or bool(net.get("OPF_converged", False)):
                opf_time_sum += (t1 - t0)
                opf_time_kept += 1
            costs.append(evaluate_total_cost(net))
            summary = _contingency_eval(net, contingencies, vm_bus_idx, ctol, {"enforce_q_lims": True})
            base_hard_feasible_flags.append(summary["base_feasible"])
            hard_feasible_flags.append(summary["sc_feasible"])
            #print(f'feas arr: {hard_feasible_flags}')
            base_penalties.append(summary["base_penalty"])
            max_contingency_penalties.append(summary["max_contingency_penalty"])
            sc_penalties.append(summary["sc_penalty"])
            failed_contingency_counts.append(summary["failed_contingency_count"])
        except Exception as e:
            if verbose:
                print(f"[WARN] OPF or metrics failed on {path}: {e}")
            costs.append(float("nan"))
            base_hard_feasible_flags.append(False)
            hard_feasible_flags.append(False)
            base_penalties.append(float(gvars.max_pen))
            max_contingency_penalties.append(float(gvars.max_pen))
            sc_penalties.append(float(gvars.max_pen))
            failed_contingency_counts.append(len(contingencies))

    avg_cost = float(np.mean([c for c in costs if np.isfinite(c)])) if costs else float("nan")
    avg_opf_solve_time = (opf_time_sum / opf_time_kept) if opf_time_kept > 0 else float("nan")
    return {
        "n_samples": n_samples,
        "costs": costs,
        "avg_cost": avg_cost,
        "base_hard_feasible_flags": base_hard_feasible_flags,
        "hard_feasible_flags": hard_feasible_flags,
        "base_feasibility_rate_%": 100.0 * float(np.mean(base_hard_feasible_flags)) if base_hard_feasible_flags else float("nan"),
        "feasibility_rate_%": 100.0 * float(np.mean(hard_feasible_flags)) if hard_feasible_flags else float("nan"),
        "base_penalties": base_penalties,
        "max_contingency_penalties": max_contingency_penalties,
        "sc_penalties": sc_penalties,
        "failed_contingency_counts": failed_contingency_counts,
        "infer_time_per_sample_s": avg_opf_solve_time,
    }


class PFEvaluatorSC:
    """Baseline independent AC SC evaluator."""

    def __init__(
        self,
        denorm_fn: Callable[[Any, Dict[str, Any], str], Any],
        apply_pred_fn: Callable[[pp.pandapowerNet, Dict[str, Any], Any, Any, Callable, str], pp.pandapowerNet],
        model_infer_fn: Callable[[Any, Any, str], Any] | None = None,
        ctol: float = gvars.ctol,
        runpp_kwargs: Dict[str, Any] | None = None,
    ):
        self.denorm_fn = denorm_fn
        self.apply_pred_fn = apply_pred_fn
        self.model_infer_fn = model_infer_fn
        self.ctol = float(ctol)
        self.runpp_kwargs = dict(enforce_q_lims=True)
        if runpp_kwargs:
            self.runpp_kwargs.update(runpp_kwargs)

    def _is_torch(self, obj) -> bool:
        return (torch is not None) and hasattr(obj, "__class__") and "torch" in str(obj.__class__.__module__)

    def _default_model_infer(self, model: Any, xb: Any, device: str):
        if self._is_torch(model):
            xb_t = xb.to(device) if hasattr(xb, "to") else xb
            return model(xb_t)
        return model(xb)

    def evaluate(
        self,
        model: Any,
        loader: Iterable[Tuple[Any, Any]],
        meta: Dict[str, Any],
        net_paths: List[str],
        contingency_spec=None,
        device: str = "cpu",
        collect_norm_mse: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        n_samples = 0
        costs: List[float] = []
        base_hard_feasible_flags: List[bool] = []
        hard_feasible_flags: List[bool] = []
        base_penalties: List[float] = []
        max_contingency_penalties: List[float] = []
        sc_penalties: List[float] = []
        failed_contingency_counts: List[int] = []
        mse_norm_sum = 0.0
        use_torch = (torch is not None) and self._is_torch(model)
        mse_loss = nn.MSELoss(reduction="mean") if (collect_norm_mse and use_torch) else None
        infer_time_sum = 0.0
        infer_kept_count = 0
        net_path_ind = 0

        for xb, yb in loader:
            t0 = time.perf_counter()
            y_pred = self.model_infer_fn(model, xb, device) if self.model_infer_fn is not None else self._default_model_infer(model, xb, device)
            batch_infer_time = time.perf_counter() - t0
            if collect_norm_mse and use_torch:
                yb_t = yb.to(device).float()
                mse_norm_sum += float(mse_loss(y_pred, yb_t).item()) * yb_t.size(0)
            B = y_pred.shape[0] if hasattr(y_pred, "shape") else len(y_pred)
            per_sample_infer = batch_infer_time / max(1, B)

            for i in range(B):
                n_samples += 1
                path = net_paths[net_path_ind]
                net_path_ind += 1
                try:
                    loaded_net = util.load_net(path)
                    prep_constraints(loaded_net)
                    vm_bus_idx = get_orig_pq_bus_idx(loaded_net)
                    contingencies = filter_valid_contingencies(loaded_net, resolve_contingencies(loaded_net, contingency_spec=contingency_spec, scenario_path=path))
                    x_i = xb[i] if hasattr(xb, "__getitem__") else xb
                    y_i = y_pred[i] if hasattr(y_pred, "__getitem__") else y_pred
                    net_i = self.apply_pred_fn(copy.deepcopy(loaded_net), meta, x_i, y_i, self.denorm_fn, device)
                    run_pf(net_i, enforce_q_lims=self.runpp_kwargs.get("enforce_q_lims", True))
                    costs.append(evaluate_total_cost(net_i))
                    summary = _contingency_eval(net_i, contingencies, vm_bus_idx, self.ctol, self.runpp_kwargs)
                    base_hard_feasible_flags.append(summary["base_feasible"])
                    #print(f'feas arr: {base_hard_feasible_flags}')
                    hard_feasible_flags.append(summary["sc_feasible"])
                    base_penalties.append(summary["base_penalty"])
                    max_contingency_penalties.append(summary["max_contingency_penalty"])
                    sc_penalties.append(summary["sc_penalty"])
                    failed_contingency_counts.append(summary["failed_contingency_count"])
                    infer_time_sum += per_sample_infer
                    infer_kept_count += 1
                except Exception as e:
                    if verbose:
                        print(f"[WARN] SC evaluation failed on sample {i}: {e}")
                    costs.append(float("nan"))
                    base_hard_feasible_flags.append(False)
                    hard_feasible_flags.append(False)
                    base_penalties.append(float(gvars.max_pen))
                    max_contingency_penalties.append(float(gvars.max_pen))
                    sc_penalties.append(float(gvars.max_pen))
                    failed_contingency_counts.append(0)

        results = {
            "n_samples": n_samples,
            "costs": costs,
            "avg_cost": float(np.nanmean(costs)) if costs else float("nan"),
            "base_hard_feasible_flags": base_hard_feasible_flags,
            "hard_feasible_flags": hard_feasible_flags,
            "base_feasibility_rate_%": 100.0 * float(np.mean(base_hard_feasible_flags)) if base_hard_feasible_flags else float("nan"),
            "feasibility_rate_%": 100.0 * float(np.mean(hard_feasible_flags)) if hard_feasible_flags else float("nan"),
            "base_penalties": base_penalties,
            "max_contingency_penalties": max_contingency_penalties,
            "sc_penalties": sc_penalties,
            "failed_contingency_counts": failed_contingency_counts,
            "infer_time_per_sample_s": (infer_time_sum / infer_kept_count) if infer_kept_count > 0 else float("nan"),
            "infer_time_kept_n": infer_kept_count,
        }
        if collect_norm_mse and n_samples > 0:
            results["mse_norm"] = mse_norm_sum / n_samples
        return results
