'''
This evaluator checks predicted generator dispatch against a prebuilt
DC optimization model of the scenario. It does NOT rerun a full
pandapower PF/OPF after every prediction. Rather, uses the linear
DC representation of the network to evaluate feasibility, cost, and
constraint violations efficiently.

Steps:
------
Load one scenario net
Convert to DC case
Get contingency rows with `resolve_contingency_rows(...)`
Build an SC-aware DC case with `build_dc_case(...)`--> includes contingency cases
Convert model output into generator dispatch, pg
Check feasibility of that pg on the SC case--> check on not only the base case but also on contingency cases
Optionally project infeasible pg back to a feasible one.
Record cost, feasibility, penalty, runtime, balance error, and max flow violation.
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


class DCSCEvaluator:
    """
    Generic evaluator for DC / DC-SC generator-dispatch baselines.

    Parameters
    ----------
    denorm_fn : Callable
        Function that converts normalized model outputs into physical units,
        when needed by the baseline. The evaluator itself does not assume a
        specific prediction format; it simply passes this into apply_pred_fn.

    apply_pred_fn : Callable
        Adapter function that turns the model output for one sample into the
        final generator dispatch vector `pg`.

        Expected signature:
            apply_pred_fn(dc_case, meta, x_i, ypred_i, denorm_fn, device="cpu")

        Expected return:
            dict with key:
                "pg" : array-like, shape (ng,)
                    Generator dispatch in MW.

        This is the key hook that makes the evaluator reusable across:
        - neural baselines
        - optimization baselines
        - hybrids that reconstruct missing generators, denormalize outputs, etc.

    model_infer_fn : Callable, optional
        Optional custom inference function for the model.

        Expected signature:
            model_infer_fn(model, xb, device="cpu") -> ypred_batch

        Use this when the baseline does not simply do `model(xb)`, e.g. if it
        reconstructs balancing generation internally or needs a custom forward
        path during evaluation.

    feas_tol : float
        Tolerance used in DC feasibility checks.

    max_penalty : float
        Penalty cap used when recording failed/infeasible samples.

    ctol : float
        Reserved numerical tolerance. Kept for compatibility with the broader
        evaluation interface.

    project_if_needed : bool
        If True, infeasible predicted dispatches are passed through
        `project_pg_to_feasible(...)` before final scoring.
    """
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
        """
        Evaluate a baseline on a list of scenario nets.

        Parameters
        ----------
        model : torch.nn.Module or compatible object
            The model/baseline object. It can be a real neural network, or even
            a dummy placeholder if all real work happens inside apply_pred_fn.

        loader : DataLoader
            Batch loader for input/target tensors.

            Important:
            - len(loader.dataset) must exactly match len(net_paths)
            - sample order must match the order of net_paths

        meta : dict
            Baseline metadata/configuration used during evaluation.

        net_paths : sequence of str
            Paths to scenario nets. One path per sample.

        device : str
            "cpu" or "cuda". Used for model inference.

        collect_norm_mse : bool
            If True, also compute normalized-space MSE between predicted output
            and target output from the loader.

            This is mostly useful for ML baselines. For optimization baselines
            using dummy outputs, this metric is usually not meaningful.

        verbose : bool
            If True, print the final feasibility rate.

        Returns
        -------
        out : dict
            Dictionary containing per-sample and aggregate evaluation results.
        """
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

                        cont_rows = resolve_contingency_rows(  #which rows to use for building contingency cases
                            net,
                            ppc,
                            contingency_spec=meta.get("contingency_rows"),
                            scenario_path=path,
                        )

                        dc_case = build_dc_case(ppc, contingency_rows=cont_rows)

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