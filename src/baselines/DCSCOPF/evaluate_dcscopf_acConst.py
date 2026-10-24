'''
Use this file to evaluate SC DC-OPF solvers using AC constraints. 
Just to check if they satisfy AC constraints or not 
'''
from __future__ import annotations

import argparse
import copy
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.pypower.idx_gen import GEN_BUS, VG

import src.global_vars as gvars
from src import util
from src.dc_utils import (
    aggregate_bus_pd_from_ppc,
    build_dc_case,
    get_dc_ppc,
    resolve_contingency_rows,
    solve_sc_dcopf,
)
from src.global_evaluation_AC import (
    compute_penalty_from_net,
    evaluate_total_cost,
    get_orig_pq_bus_idx,
    is_feasible_by_hard_constraints,
)


def _read_manifest_paths(case: str, which_data: str) -> List[str]:
    """
    Read valid scenario file paths for one test split.
    """
    scen_dir = os.path.join("src", "scenarios", case, "test", which_data)
    manifest_path = os.path.join(scen_dir, "manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    return [os.path.join(scen_dir, f) for f in manifest["file"].tolist()]


def _get_pp_bus_to_ppc_bus_lookup(net: pp.pandapowerNet) -> Dict[int, int]:
    """
    Get mapping: pandapower bus index -> ppc bus row index.

    Assumes get_dc_ppc(net) has already been called on this same net object.
    """
    lookups = net.get("_pd2ppc_lookups", {})
    if "bus" not in lookups:
        raise RuntimeError("Could not find pandapower bus lookup on net['_pd2ppc_lookups'].")

    bus_lookup = lookups["bus"]
    out: Dict[int, int] = {}

    if isinstance(bus_lookup, dict):
        for k, v in bus_lookup.items():
            if v is not None and int(v) >= 0:
                out[int(k)] = int(v)
        return out

    arr = np.asarray(bus_lookup)
    for b in net.bus.index.to_numpy(dtype=int):
        if 0 <= int(b) < len(arr):
            ppc_b = int(arr[int(b)])
            if ppc_b >= 0:
                out[int(b)] = ppc_b
    return out


def _build_ppc_gen_row_mapping(net: pp.pandapowerNet, ppc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Map each ppc generator row back to the corresponding pandapower source row
    (either ext_grid or gen).

    Needed because solve_sc_dcopf(...) returns pg in ppc generator order,
    while AC PF evaluation needs values applied back on the pandapower net.
    """
    bus_map = _get_pp_bus_to_ppc_bus_lookup(net)

    ppc_gen = np.asarray(ppc["gen"], dtype=float)
    ppc_gen_bus = ppc_gen[:, GEN_BUS].astype(int)
    ppc_gen_vg = ppc_gen[:, VG].astype(float)

    candidates: List[Dict[str, Any]] = []

    if len(net.ext_grid):
        eg = net.ext_grid
        eg_on = eg["in_service"].fillna(True).to_numpy().astype(bool) if "in_service" in eg.columns else np.ones(len(eg), dtype=bool)
        for idx in eg.index[eg_on].tolist():
            bus = int(eg.at[idx, "bus"])
            candidates.append(
                {
                    "kind": "ext_grid",
                    "pp_idx": int(idx),
                    "pp_bus": bus,
                    "ppc_bus": int(bus_map[bus]),
                    "vm_pu": float(eg.at[idx, "vm_pu"]) if "vm_pu" in eg.columns else 1.0,
                }
            )

    if len(net.gen):
        gen = net.gen
        gen_on = gen["in_service"].fillna(True).to_numpy().astype(bool) if "in_service" in gen.columns else np.ones(len(gen), dtype=bool)
        for idx in gen.index[gen_on].tolist():
            bus = int(gen.at[idx, "bus"])
            candidates.append(
                {
                    "kind": "gen",
                    "pp_idx": int(idx),
                    "pp_bus": bus,
                    "ppc_bus": int(bus_map[bus]),
                    "vm_pu": float(gen.at[idx, "vm_pu"]) if "vm_pu" in gen.columns else 1.0,
                }
            )

    row_map: Dict[int, Dict[str, Any]] = {}
    used_candidate_idx = set()

    for r in range(ppc_gen.shape[0]):
        target_bus = int(ppc_gen_bus[r])
        target_vg = float(ppc_gen_vg[r])

        best_idx = None
        best_score = None

        for cand_i, cand in enumerate(candidates):
            if cand_i in used_candidate_idx:
                continue
            if cand["ppc_bus"] != target_bus:
                continue

            score = abs(cand["vm_pu"] - target_vg)
            if cand["kind"] == "ext_grid":
                score -= 1e-9  # tie-break preference

            if best_score is None or score < best_score:
                best_score = score
                best_idx = cand_i

        if best_idx is None:
            raise RuntimeError(f"Could not map ppc gen row {r} back to pandapower source.")

        used_candidate_idx.add(best_idx)
        row_map[r] = candidates[best_idx]

    return row_map


def _apply_dc_pg_to_ac_net(
    net: pp.pandapowerNet,
    pg: np.ndarray,
    row_map: Dict[int, Dict[str, Any]],
) -> None:
    """
    Apply solved DC dispatch back to the pandapower AC net.

    Important:
    - For 'gen' rows, set net.gen.p_mw directly.
    - For 'ext_grid', do nothing. In AC PF, ext_grid active power is solved as slack.
    """
    pg = np.asarray(pg, dtype=float).reshape(-1)

    for r, src in row_map.items():
        if src["kind"] == "gen":
            net.gen.at[src["pp_idx"], "p_mw"] = float(pg[r])


def _evaluate_one_path(
    path: str,
    contingency_spec: Optional[str],
    warm_start_from_base_pg: bool,
) -> Dict[str, Any]:
    """
    1) Solve iterative DC-SCOPF on the scenario
    2) Apply the resulting dispatch to the AC pandapower net
    3) Run AC PF
    4) Evaluate feasibility with global_evaluation_AC.py helpers
    """
    t0 = time.perf_counter()
    max_penalty = float(getattr(gvars, "max_pen", 1e6))

    try:
        # Load once, then make separate copies for DC solve and AC evaluation.
        loaded_net = util.load_net(path)

        dc_net = copy.deepcopy(loaded_net)
        ac_net = copy.deepcopy(loaded_net)

        # ---------- DC solve ----------
        ppc = get_dc_ppc(dc_net)

        cont_rows = resolve_contingency_rows(
            dc_net,
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

        # ---------- AC evaluation ----------
        row_map = _build_ppc_gen_row_mapping(dc_net, ppc)

        ac_net["_orig_pq_bus_idx"] = get_orig_pq_bus_idx(ac_net)
        _apply_dc_pg_to_ac_net(ac_net, pg, row_map)

        pp.runpp(
            ac_net,
            enforce_q_lims=True,
            calculate_voltage_angles=True,
        )

        feasible = bool(
            is_feasible_by_hard_constraints(
                ac_net,
                vm_bus_idx=ac_net["_orig_pq_bus_idx"],
            )
        )

        cost = float(evaluate_total_cost(ac_net))
        penalty = float(compute_penalty_from_net(ac_net))

        return {
            "path": path,
            "file": os.path.basename(path),
            "success": True,
            "solver_success": bool(solve_out.get("success", False)),
            "solver_status": solve_out.get("status", None),
            "solver_message": str(solve_out.get("message", "")),
            "feasible": feasible,
            "cost": cost,
            "penalty": penalty,
            "runtime_s": float(time.perf_counter() - t0),
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
            "error": f"{type(e).__name__}: {e}",
        }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    costs = np.asarray([r["cost"] for r in rows], dtype=float)
    feasible = np.asarray([r["feasible"] for r in rows], dtype=bool)
    penalties = np.asarray([r["penalty"] for r in rows], dtype=float)
    runtimes = np.asarray([r["runtime_s"] for r in rows], dtype=float)

    finite_costs = costs[np.isfinite(costs)]

    return {
        "n_samples": int(len(rows)),
        "hard_feasible_flags": feasible,
        "feasibility_rate_%": float(100.0 * feasible.mean()) if len(feasible) else 0.0,
        "costs": costs,
        "avg_cost": float(np.mean(finite_costs)) if finite_costs.size else float("nan"),
        "penalties": penalties,
        "avg_penalty": float(np.mean(penalties)) if len(penalties) else float("nan"),
        "runtimes_s": runtimes,
        "infer_time_per_sample_s": float(np.mean(runtimes)) if len(runtimes) else float("nan"),
        "rows": rows,
    }


def generate_iterative_dcscopf_ac_eval_result(
    case: str,
    which_data: str,
    contingency_spec: Optional[str] = None,
    warm_start_from_base_pg: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate iterative DC-SCOPF solutions using AC PF + AC hard constraints.
    """
    net_paths = _read_manifest_paths(case, which_data)
    rows: List[Dict[str, Any]] = []

    for i, path in enumerate(net_paths, start=1):
        row = _evaluate_one_path(
            path=path,
            contingency_spec=contingency_spec,
            warm_start_from_base_pg=warm_start_from_base_pg,
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

    res = _aggregate_rows(rows)
    res["case"] = case
    res["which_data"] = which_data
    res["solver_name"] = "iterative_dc_scopf"
    res["evaluation_mode"] = "ac_constraints"
    res["contingency_spec"] = contingency_spec
    res["warm_start_from_base_pg"] = bool(warm_start_from_base_pg)

    out_dir = os.path.join("results", case)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"iterative_dcscopf_ac_eval_result_{which_data}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved AC-evaluated iterative DC-SCOPF results to {out_path}")
    print(f"Feasibility rate (%): {res['feasibility_rate_%']:.2f}")
    print(f"Avg runtime per sample (s): {res['infer_time_per_sample_s']:.6f}")
    print(f"Avg cost: {res['avg_cost']:.6f}")

    return res


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate iterative DC-SCOPF baseline on AC constraints."
    )
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument(
        "--which_data",
        type=str,
        default="all",
        help="One test split name, or 'all' for all test splits.",
    )
    parser.add_argument(   #for the dcsc solver, which contingency cases we want to have?
        "--contingencies",
        type=str,
        default=None,  #consider all branches for DC SC
        help=(
            "Passed into resolve_contingency_rows(...).\n"
            "Examples:\n"
            "  unset / None -> sidecar if present, else all in-service branches\n"
            "  \"[]\"        -> no contingencies, just base dc case\n"
            "  \"[0,1,2]\"   -> explicit ppc branch rows"
        ),
    )
    parser.add_argument(
        "--no_warm_start_from_base_pg",
        action="store_true",
        help="Disable warm start from dc_case['base_pg'] before solve_sc_dcopf().",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    which_case_list = list(gvars.stress_cases)
    which_case_list.append("in_distribution")

    if args.which_data != "all":
        which_case_list = [args.which_data]

    for which_data in which_case_list:
        print(f"\n=== Evaluating iterative DC-SCOPF on {which_data} with AC constraints ===")
        generate_iterative_dcscopf_ac_eval_result(
            case=args.case,
            which_data=which_data,
            contingency_spec=args.contingencies,
            warm_start_from_base_pg=(not args.no_warm_start_from_base_pg),
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()