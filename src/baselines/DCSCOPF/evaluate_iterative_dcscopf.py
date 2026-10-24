import argparse
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import src.global_vars as gvars
import src.global_vars_dc as gvars_dc
from src import util
from src.dc_utils import (
    aggregate_bus_pd_from_ppc,
    build_dc_case,
    compute_cost,
    feasibility_summary,
    get_dc_ppc,
    resolve_contingency_rows,
    solve_sc_dcopf,
)


def _limit_threads_per_process(math_threads: int = 1) -> None:
    """
    Limit BLAS / OpenMP thread usage inside each worker process.

    ---
    Each scenario solve is already parallelized at the process level by
    ProcessPoolExecutor. If every worker also uses many BLAS threads, the
    machine can oversubscribe badly and become slower.

    Parameters
    ----------
    math_threads : int
        Number of math-library threads per worker.
    """
    value = str(max(1, int(math_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = value


def _read_manifest_paths(case: str, which_data: str) -> List[str]:
    """
    Read the scenario manifest for one test split and return valid net paths.

    Parameters
    ----------
    case : str
        Case name, e.g. "case14".
    which_data : str
        Test split name, e.g. "in_distribution", "load_extreme_high".

    Returns
    -------
    list[str]
        Full paths to scenario files whose manifest status is "ok".
    """
    scen_dir = os.path.join("src", "scenarios", case, "test", which_data)
    manifest_path = os.path.join(scen_dir, "manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    return [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]


def _evaluate_one_path(
    path: str,
    contingency_spec: Optional[str],
    warm_start_from_base_pg: bool,
    feas_tol: float,
    max_penalty: float,
) -> Dict[str, Any]:
    """
    Solve and evaluate one scenario independently.

    Workflow
    --------
    1) load pandapower net
    2) convert to DC PPC
    3) resolve contingency rows
    4) build SC-aware DC case
    5) solve iterative DC-SCOPF with solve_sc_dcopf(...)
    6) evaluate feasibility / penalty / cost

    Parameters
    ----------
    path : str
        Path to one saved scenario net.
    contingency_spec : str or None
        Passed into resolve_contingency_rows(...).
        Examples:
        - None  -> sidecar if present, else all in-service branches
        - "[]"  -> no contingencies (base-case-only DCOPF)
        - "[0,1,2]" -> explicit PPC branch rows
    warm_start_from_base_pg : bool
        If True, initialize solve_sc_dcopf(...) with dc_case["base_pg"].
    feas_tol : float
        Feasibility tolerance used by feasibility_summary(...).
    max_penalty : float
        Penalty value used for failed samples.

    Returns
    -------
    dict
        Per-sample result record with keys such as:
        - path
        - success
        - solver_success
        - feasible
        - cost
        - penalty
        - runtime_s
        - balance_error
        - max_flow_violation
        - error
    """
    t0 = time.perf_counter()

    try:
        net = util.load_net(path)
        ppc = get_dc_ppc(net)

        cont_rows = resolve_contingency_rows(
            net,
            ppc,
            contingency_spec=contingency_spec,
            scenario_path=path,
        )

        dc_case = build_dc_case(ppc, contingency_rows=cont_rows)
        pd_bus = aggregate_bus_pd_from_ppc(ppc)

        x0 = None
        if warm_start_from_base_pg:
            x0 = np.asarray(dc_case["base_pg"], dtype=float).reshape(-1)

        solve_out = solve_sc_dcopf(dc_case, pd_bus, x0=x0)

        pg = np.asarray(solve_out["pg"], dtype=float).reshape(-1)
        feas = feasibility_summary(dc_case, pg, pd_bus, tol=feas_tol)
        cost = float(compute_cost(dc_case, pg))

        return {
            "path": path,
            "file": os.path.basename(path),
            "success": True,
            "solver_success": bool(solve_out.get("success", False)),
            "solver_status": solve_out.get("status", None),
            "solver_message": str(solve_out.get("message", "")),
            "feasible": bool(feas["feasible"]),
            "cost": cost,
            "penalty": float(min(feas["penalty"], max_penalty)),
            "runtime_s": float(time.perf_counter() - t0),
            "used_projection": False,  # raw iterative solver only
            "balance_error": float(feas["balance_error"]),
            "max_flow_violation": float(feas["max_flow_violation"]),
            "error": "",
        }

    except Exception as e:
        return {
            "path": path,
            "file": os.path.basename(path),
            "success": False,
            "solver_success": False,
            "solver_status": None,
            "solver_message": "",
            "feasible": False,
            "cost": float("nan"),
            "penalty": float(max_penalty),
            "runtime_s": float(time.perf_counter() - t0),
            "used_projection": False,
            "balance_error": float(max_penalty),
            "max_flow_violation": float(max_penalty),
            "error": f"{type(e).__name__}: {e}",
        }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert per-sample row records into the same style of aggregate output
    used by the earlier evaluator.

    Parameters
    ----------
    rows : list[dict]
        One dict per scenario.

    Returns
    -------
    dict
        Aggregate evaluation result containing arrays and summary statistics.
    """
    costs_arr = np.asarray([r["cost"] for r in rows], dtype=float)
    feas_arr = np.asarray([r["feasible"] for r in rows], dtype=bool)
    pen_arr = np.asarray([r["penalty"] for r in rows], dtype=float)
    run_arr = np.asarray([r["runtime_s"] for r in rows], dtype=float)
    used_proj_arr = np.asarray([r["used_projection"] for r in rows], dtype=bool)
    bal_arr = np.asarray([r["balance_error"] for r in rows], dtype=float)
    flow_arr = np.asarray([r["max_flow_violation"] for r in rows], dtype=float)

    finite_costs = costs_arr[np.isfinite(costs_arr)]

    return {
        "n_samples": int(len(rows)),
        "hard_feasible_flags": feas_arr,
        "feasibility_rate_%": float(100.0 * feas_arr.mean()) if len(feas_arr) else 0.0,
        "costs": costs_arr,
        "avg_cost": float(np.mean(finite_costs)) if finite_costs.size else float("nan"),
        "penalties": pen_arr,
        "avg_penalty": float(np.mean(pen_arr)) if len(pen_arr) else float("nan"),
        "runtimes_s": run_arr,
        "infer_time_per_sample_s": float(np.mean(run_arr)) if len(run_arr) else float("nan"),
        "used_projection": used_proj_arr,
        "balance_errors": bal_arr,
        "max_flow_violation": flow_arr,
        "rows": rows,
    }


def generate_iterative_dcscopf_result(
    case: str,
    which_data: str,
    contingency_spec: Optional[str] = None,
    warm_start_from_base_pg: bool = True,
    verbose: bool = False,
    num_workers: int = 1,
    math_threads_per_worker: int = 1,
    mp_start_method: Optional[str] = None,
):
    """
    Evaluate the iterative DC-SCOPF solver on one test split and save results.

    for faster computation, parallelizes over scenario files.
    
    """

    net_paths = _read_manifest_paths(case, which_data)
    feas_tol = float(getattr(gvars_dc, "dc_balance_tol_mw", 1e-6))
    max_penalty = float(getattr(gvars_dc, "dc_max_pen", 1e6))

    rows: List[Dict[str, Any]] = []

    num_workers = max(1, min(int(num_workers), len(net_paths))) if len(net_paths) else 1

    if num_workers == 1:
        for i, path in enumerate(net_paths, start=1):
            row = _evaluate_one_path(
                path=path,
                contingency_spec=contingency_spec,
                warm_start_from_base_pg=bool(warm_start_from_base_pg),
                feas_tol=feas_tol,
                max_penalty=max_penalty,
            )
            rows.append(row)

            if verbose:
                print(
                    f"[{i}/{len(net_paths)}] {row['file']} "
                    f"feasible={row['feasible']} "
                    f"solver_success={row['solver_success']} "
                    f"runtime_s={row['runtime_s']:.4f}",
                    flush=True,
                )
    else:
        if mp_start_method is None:
            mp_start_method = "fork" if ("fork" in mp.get_all_start_methods()) else "spawn"

        ctx = mp.get_context(mp_start_method)

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_limit_threads_per_process,
            initargs=(math_threads_per_worker,),
        ) as ex:
            future_to_path = {
                ex.submit(
                    _evaluate_one_path,
                    path,
                    contingency_spec,
                    bool(warm_start_from_base_pg),
                    feas_tol,
                    max_penalty,
                ): path
                for path in net_paths
            }

            for i, fut in enumerate(as_completed(future_to_path), start=1):
                row = fut.result()
                rows.append(row)

                if verbose:
                    print(
                        f"[{i}/{len(net_paths)}] {row['file']} "
                        f"feasible={row['feasible']} "
                        f"solver_success={row['solver_success']} "
                        f"runtime_s={row['runtime_s']:.4f}",
                        flush=True,
                    )

        # Restore original manifest order in the final saved result.
        order = {p: i for i, p in enumerate(net_paths)}
        rows.sort(key=lambda r: order[r["path"]])

    res = _aggregate_rows(rows)
    res["case"] = case
    res["which_data"] = which_data
    res["solver_name"] = "iterative_dc_scopf"
    res["contingency_spec"] = contingency_spec
    res["warm_start_from_base_pg"] = bool(warm_start_from_base_pg)
    res["num_workers"] = int(num_workers)
    res["math_threads_per_worker"] = int(math_threads_per_worker)

    out_dir = os.path.join("results", case)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"iterative_dcscopf_result_{which_data}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved iterative DC-SCOPF results to {out_path}")
    print(f"Feasibility rate (%): {res['feasibility_rate_%']:.2f}")
    print(f"Avg runtime per sample (s): {res['infer_time_per_sample_s']:.6f}")
    print(f"Avg cost: {res['avg_cost']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate iterative DC-SC-OPF solver on test scenarios and save results."
    )
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument(
        "--which_data",
        type=str,
        default="all",
        help="One test split name, or 'all' for all test splits.",
    )
    parser.add_argument(
        "--contingencies",
        type=str,
        default=None,
        help=(
            "Optional contingency spec passed into resolve_contingency_rows(...).\n"
            "Examples:\n"
            "  unset / None -> sidecar if present, else all in-service branches\n"
            "  \"[]\"        -> no contingencies (base-case-only DCOPF)\n"
            "  \"[0,1,2]\"   -> explicit PPC branch rows"
        ),
    )
    parser.add_argument(
        "--no_warm_start_from_base_pg",
        action="store_true",
        help="Disable warm start from dc_case['base_pg'] before solve_sc_dcopf().",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--math_threads_per_worker", type=int, default=1)
    parser.add_argument(
        "--mp_start_method",
        type=str,
        default=None,
        choices=[None, "fork", "spawn", "forkserver"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Also limit threads in the parent process.
    _limit_threads_per_process(args.math_threads_per_worker)

    which_case_list = list(gvars.stress_cases)
    which_case_list.append("in_distribution")

    if args.which_data != "all":
        which_case_list = [args.which_data]

    for which_data in which_case_list:
        print(f"\n=== Evaluating iterative DC-SCOPF on {which_data} ===")
        generate_iterative_dcscopf_result(
            case=args.case,
            which_data=which_data,
            contingency_spec=args.contingencies,
            warm_start_from_base_pg=(not args.no_warm_start_from_base_pg),
            verbose=args.verbose,
            num_workers=args.num_workers,
            math_threads_per_worker=args.math_threads_per_worker,
            mp_start_method=args.mp_start_method,
        )


if __name__ == "__main__":
    main()