'''
Feasibility is checked only on base case.
Steps:
------
Same as `global_evaluation_DCSC` except `build_dc_case(ppc, contingency_rows=[])`
now uses [] as no contingency is being evaluated.
'''

import time
from typing import Any, Callable, Dict, Optional
import traceback
import numpy as np
import torch

from src import util
from src.dc_utils import (
    aggregate_bus_pd_from_ppc,
    build_dc_case,
    compute_cost,
    feasibility_summary,
    get_dc_ppc,
    project_pg_to_feasible,
    reconstruct_theta,
    resolve_contingency_rows,
)


class DCEvaluator:
    def __init__(
        self,
        denorm_fn: Callable,
        apply_pred_fn: Callable,
        model_infer_fn: Optional[Callable] = None,
        feas_tol: float = 1e-6,
        max_penalty: float = 1e6,
        ctol: float = 1e-7,
        project_if_needed: bool = True,
    ):
        self.denorm_fn = denorm_fn
        self.apply_pred_fn = apply_pred_fn
        self.model_infer_fn = model_infer_fn
        self.feas_tol = float(feas_tol)
        self.max_penalty = float(max_penalty)
        self.ctol = float(ctol)
        self.project_if_needed = bool(project_if_needed)

    def evaluate(
        self,
        model,
        loader,
        meta: Dict[str, Any],
        net_paths,
        device: str = "cpu",
        collect_norm_mse: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        if hasattr(model, "eval"):
            model.eval()

        net_paths = list(net_paths)
        if len(net_paths) != len(loader.dataset):
            raise ValueError(
                f"len(net_paths)={len(net_paths)} does not match len(loader.dataset)={len(loader.dataset)}"
            )

        costs = []
        feasible = []
        penalties = []
        runtimes = []
        used_projection = []
        balance_errors = []
        max_flow_viol = []
        norm_mse_terms = []

        # ensure these are numpy arrays 
        #x_sd = np.asarray(meta["x_sd_pu"], dtype=float)
        #x_mu = np.asarray(meta["x_mu_pu"], dtype=float)

        path_idx = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(dev).float()
                yb = yb.to(dev).float()

                if self.model_infer_fn is not None:
                    t0 = time.perf_counter()
                    ypred = self.model_infer_fn(model, xb, device=str(dev))
                    infer_elapsed = time.perf_counter() - t0
                    ypred = ypred.to(dev).float()
                else:
                    t0 = time.perf_counter()
                    ypred = model(xb)
                    infer_elapsed = time.perf_counter() - t0
                    ypred = ypred.to(dev).float()

                if collect_norm_mse:
                    norm_mse_terms.append(float(torch.mean((ypred - yb) ** 2).detach().cpu()))

                for i in range(xb.shape[0]):
                    path = net_paths[path_idx]
                    path_idx += 1
                    sample_t0 = time.perf_counter()

                    try:
                        net = util.load_net(path)

                        ppc = get_dc_ppc(net)  #get pypower case file

                        dc_case = build_dc_case(ppc, contingency_rows=[]) #use [] so it's non-sc evaluation

                        pd_bus = aggregate_bus_pd_from_ppc(ppc)

                        pred_obj = self.apply_pred_fn(
                            dc_case,
                            meta,
                            xb[i],
                            ypred[i],
                            self.denorm_fn,
                            device=str(dev),
                        )

                        pg = pred_obj["pg"]
                        if isinstance(pg, torch.Tensor):  # robustness
                            pg = pg.detach().cpu().numpy()
                        pg = np.asarray(pg, dtype=float).reshape(-1)

                        feas = feasibility_summary(dc_case, pg, pd_bus, tol=self.feas_tol)

                        #x_raw_i = (xb[i].detach().cpu().numpy() * x_sd + x_mu).astype(float)

                        projected = False
                        if (not feas["feasible"]) and self.project_if_needed:
                            proj = project_pg_to_feasible(dc_case, pd_bus, pg)
                            if proj["success"]:
                                pg = np.asarray(proj["pg"], dtype=float).reshape(-1)
                                feas = feasibility_summary(dc_case, pg, pd_bus, tol=self.feas_tol)
                                projected = True

                        #theta = reconstruct_theta(dc_case, pg, pd_bus, which=0) #for debug

                        costs.append(float(compute_cost(dc_case, pg)))
                        feasible.append(bool(feas["feasible"]))
                        penalties.append(float(min(feas["penalty"], self.max_penalty)))
                        used_projection.append(bool(projected))
                        balance_errors.append(float(feas["balance_error"]))
                        max_flow_viol.append(float(feas["max_flow_violation"]))

                        #  append runtime on successful samples
                        runtimes.append(
                            float(infer_elapsed / max(1, xb.shape[0]) + (time.perf_counter() - sample_t0))
                        )

                    except Exception as e:
                        print(f"[ERROR] path={path}")
                        print(f"[ERROR] type={type(e).__name__}: {e}")
                        traceback.print_exc()
                        costs.append(float("nan"))
                        feasible.append(False)
                        penalties.append(float(self.max_penalty))
                        used_projection.append(False)
                        balance_errors.append(float(self.max_penalty))
                        max_flow_viol.append(float(self.max_penalty))
                        #not tracking runtime for failed solutions
                        #runtimes.append(
                        #    float(infer_elapsed / max(1, xb.shape[0]) + (time.perf_counter() - sample_t0))
                        #)

        costs_arr = np.asarray(costs, dtype=float)
        feas_arr = np.asarray(feasible, dtype=bool)
        pen_arr = np.asarray(penalties, dtype=float)
        run_arr = np.asarray(runtimes, dtype=float)

        finite_costs = costs_arr[np.isfinite(costs_arr)]  # avoids nanmean warning if all nan

        out = {
            "n_samples": int(len(costs_arr)),
            "hard_feasible_flags": feas_arr,
            "feasibility_rate_%": float(100.0 * feas_arr.mean()) if len(feas_arr) else 0.0,
            "costs": costs_arr,
            "avg_cost": float(np.mean(finite_costs)) if finite_costs.size else float("nan"),
            "penalties": pen_arr,
            "avg_penalty": float(np.mean(pen_arr)) if len(pen_arr) else float("nan"),
            "runtimes_s": run_arr,
            "infer_time_per_sample_s": float(np.mean(run_arr)) if len(run_arr) else float("nan"),
            "used_projection": np.asarray(used_projection, dtype=bool),
            "balance_errors": np.asarray(balance_errors, dtype=float),
            "max_flow_violation": np.asarray(max_flow_viol, dtype=float),
        }
        if collect_norm_mse and norm_mse_terms:
            out["norm_mse"] = float(np.mean(norm_mse_terms))
        if verbose:
            print(f"DC feasibility: {out['feasibility_rate_%']:.2f}%")
        return out