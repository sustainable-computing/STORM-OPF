'''
common util functions for dc opf baselines
'''
import ast
import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from pandapower.converter import to_ppc
from pandapower.pypower.idx_brch import BR_STATUS, RATE_A, F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE, PD, REF
from pandapower.pypower.idx_cost import COST, MODEL, NCOST, POLYNOMIAL
from pandapower.pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN
from pandapower.pypower.makeBdc import makeBdc


# Small constant to avoid divide-by-zero 
TINY = 1e-8


def _dense_array(x: Any) -> np.ndarray:
    """
    Convert a matrix-like object to a dense numpy float array.
    """
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=float)
    return np.asarray(x, dtype=float)


def _makeBdc_pp(
    bus: np.ndarray,
    branch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper around pypower.makeBdc with version-robust output handling

    Parameters
    ----------
    bus : np.ndarray
        PPC bus matrix.
    branch : np.ndarray
        PPC branch matrix

    Returns
    -------
    Bbus : np.ndarray
        DC bus susceptance matrix
    Bf : np.ndarray
        Linear map from bus angles to branch flows
    Pbusinj : np.ndarray
        Constant bus injection term from phase shifters / transformer effects
    Pfinj : np.ndarray
        Constant branch flow term from phase shifters / transformer effects

    Notes
    -----
    Depending on package version, makeBdc may return:
    - 4 values: Bbus, Bf, Pbusinj, Pfinj
    - 3 values: Bbus, Bf, Pbusinj
    - 2 values: Bbus, Bf

    This helper normalizes those cases into one fixed 4-output interface.
    """
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)
    out = makeBdc(bus, branch)

    if not isinstance(out, (tuple, list)):
        raise RuntimeError(f"Unsupported makeBdc return type: {type(out)}")

    if len(out) >= 4:
        Bbus, Bf, Pbusinj, Pfinj = out[:4]
    elif len(out) == 3:
        Bbus, Bf, Pbusinj = out
        Pfinj = np.zeros(branch.shape[0], dtype=float)
    elif len(out) == 2:
        Bbus, Bf = out
        Pbusinj = np.zeros(bus.shape[0], dtype=float)
        Pfinj = np.zeros(branch.shape[0], dtype=float)
    else:
        raise RuntimeError(f"Unsupported makeBdc return length: {len(out)}")

    return (
        _dense_array(Bbus),
        _dense_array(Bf),
        np.asarray(Pbusinj, dtype=float).reshape(-1),
        np.asarray(Pfinj, dtype=float).reshape(-1),
    )


def get_dc_ppc(net) -> Dict[str, np.ndarray]:
    """
    Convert a pandapower net to a PYPOWER-style PPC dictionary for DC/OPF use.
    """
    return to_ppc(
        copy.deepcopy(net),
        calculate_voltage_angles=True,
        init="flat",
        mode="opf",
        check_connectivity=True,
    )


def aggregate_bus_pd_from_ppc(ppc: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract active load at each bus from the PPC bus matrix.

    Returns
    -------
    pd_bus : np.ndarray, shape (nb,)
        Active demand at each bus in MW.
    """
    return np.asarray(ppc["bus"][:, PD], dtype=float).copy()


def build_gen_signatures(ppc: Dict[str, np.ndarray]) -> List[Tuple[int, float, float]]:
    """
    Build a lightweight generator signature list for inspection/debugging.

    Each signature is:
        (generator_bus, PMIN, PMAX)
    """
    gen = np.asarray(ppc["gen"], dtype=float)
    sigs: List[Tuple[int, float, float]] = []
    for i in range(gen.shape[0]):
        sigs.append(
            (
                int(round(gen[i, GEN_BUS])),
                round(float(gen[i, PMIN]), 6),
                round(float(gen[i, PMAX]), 6),
            )
        )
    return sigs


def extract_polynomial_cost(
    ppc: Dict[str, np.ndarray],
    ng: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract polynomial generator cost coefficients from PPC gencost.

    Returns
    -------
    a, b, c : np.ndarray
        Arrays of length ng such that generator cost is
            a_i * pg_i^2 + b_i * pg_i + c_i

    Notes
    -----
    - Only POLYNOMIAL cost rows are used.
    - If cost data is missing, all coefficients default to zero.
    - If fewer cost rows than generators exist, remaining generators keep zero cost.
    """
    gc = ppc.get("gencost", None)
    a = np.zeros(ng, dtype=float)
    b = np.zeros(ng, dtype=float)
    c = np.zeros(ng, dtype=float)

    if gc is None or len(gc) == 0:
        return a, b, c

    gc = np.asarray(gc, dtype=float)
    rows = min(ng, gc.shape[0])

    for i in range(rows):
        if int(gc[i, MODEL]) != POLYNOMIAL:
            continue

        ncost = int(round(gc[i, NCOST]))
        coeffs = np.asarray(gc[i, COST:COST + ncost], dtype=float)

        if ncost >= 3:
            a[i], b[i], c[i] = coeffs[-3], coeffs[-2], coeffs[-1]
        elif ncost == 2:
            a[i], b[i], c[i] = 0.0, coeffs[-2], coeffs[-1]
        elif ncost == 1:
            a[i], b[i], c[i] = 0.0, 0.0, coeffs[-1]

    return a, b, c


def _parse_jsonish_list(value: Any) -> Optional[List[Any]]:
    """
    Try to interpret `value` as a Python list.

    Supported cases
    ---------------
    - already a list / tuple / numpy array
    - JSON string like "[1, 2, 3]"
    - Python-literal string like "['line:0', 'line:1']"

    Returns
    -------
    list or None
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            out = json.loads(s)
            if isinstance(out, list):
                return out
        except Exception:
            pass
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return None


def resolve_contingency_rows(
    net,
    ppc: Dict[str, np.ndarray],
    contingency_spec: Optional[Sequence[Any]] = None,
    scenario_path: Optional[str] = None,
) -> List[int]:
    """
    Resolve which PPC branch rows should be treated as outage contingencies.

    This function supports both SC and non-SC use:

    Behavior
    --------
    1) contingency_spec is None:
       - try sidecar file `<scenario_path>.contingencies.json`
       - if nothing is found, default to ALL in-service branches
       => SC default mode

    2) contingency_spec is []:
       - use NO contingencies
       => base-case-only / non-SC mode

    3) contingency_spec contains explicit entries:
       - parse them into PPC branch row indices
       - use exactly those resolved rows

    Accepted item formats
    ---------------------
    - integer: direct PPC branch row
    - digit string: "5"
    - tagged string:
        "branch:5" / "ppc_branch:5"
        "line:5"
        "trafo:2"

    Returns
    -------
    list[int]
        PPC branch row indices. An empty list means no contingencies.

    Notes
    -----
    - 'line:k' assumes line rows come before transformer rows in PPC branch ordering.
    - invalid entries are ignored.
    """
    branch = np.asarray(ppc["branch"], dtype=float)
    n_branch = branch.shape[0]

    spec = _parse_jsonish_list(contingency_spec)

    if spec is None and scenario_path is not None:
        sidecar = scenario_path + ".contingencies.json"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                spec = raw.get("ppc_branch_rows") or raw.get("branch_contingency_ids")
            elif isinstance(raw, list):
                spec = raw

    if spec is None:
        return [i for i in range(n_branch) if int(round(branch[i, BR_STATUS])) == 1]

    resolved: List[int] = []
    n_line = len(getattr(net, "line", []))

    for item in spec:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if 0 <= idx < n_branch:
                resolved.append(idx)
            continue

        if isinstance(item, str):
            s = item.strip().lower()
            if s.isdigit():
                idx = int(s)
                if 0 <= idx < n_branch:
                    resolved.append(idx)
                continue

            if ":" in s:
                kind, idx_s = s.split(":", 1)
                if not idx_s.strip().isdigit():
                    continue
                raw_idx = int(idx_s.strip())
                if kind in {"branch", "ppc_branch"} and 0 <= raw_idx < n_branch:
                    resolved.append(raw_idx)
                elif kind == "line" and 0 <= raw_idx < min(n_line, n_branch):
                    resolved.append(raw_idx)
                elif kind == "trafo":
                    idx = n_line + raw_idx
                    if 0 <= idx < n_branch:
                        resolved.append(idx)

    return sorted(set(resolved))


def _apply_branch_outage(
    ppc: Dict[str, np.ndarray],
    outaged_row: Optional[int],
) -> Dict[str, np.ndarray]:
    """
    Return a copy of PPC where one branch is switched out of service.

    Parameters
    ----------
    outaged_row : int or None
        - None means base case
        - integer means that branch row is disabled
    """
    ppc_c = copy.deepcopy(ppc)
    if outaged_row is not None:
        ppc_c["branch"][outaged_row, BR_STATUS] = 0
    return ppc_c


def _is_ppc_connected(ppc_c: Dict[str, np.ndarray]) -> bool:
    """
    Check whether the active branch network is connected.
    """
    bus = np.asarray(ppc_c["bus"], dtype=float)
    branch = np.asarray(ppc_c["branch"], dtype=float)

    nb = int(bus.shape[0])
    if nb <= 1:
        return True

    adj = [set() for _ in range(nb)]
    for k in range(branch.shape[0]):
        if int(round(branch[k, BR_STATUS])) != 1:
            continue
        f = int(round(branch[k, F_BUS]))
        t = int(round(branch[k, T_BUS]))
        if 0 <= f < nb and 0 <= t < nb:
            adj[f].add(t)
            adj[t].add(f)

    starts = [i for i in range(nb) if adj[i]]
    if not starts:
        return False

    seen = set()
    stack = [starts[0]]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                stack.append(v)

    return len(seen) == nb


def _default_balance_gen_index(ppc: Dict[str, np.ndarray]) -> int:
    """
    Choose a default balancing generator index from the PPC.

    This is a generic solver helper only. It is NOT a model-prediction
    contract and should not be interpreted as saying anything about what
    a baseline predicts.

    Strategy
    --------
    - Prefer a generator located at the reference/slack bus.
    - If none exists, fall back to generator 0.
    """
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)

    ref_candidates = np.where(bus[:, BUS_TYPE] == REF)[0]
    slack_bus = int(ref_candidates[0]) if len(ref_candidates) else 0

    slack_gen_candidates = np.where(np.round(gen[:, GEN_BUS]).astype(int) == slack_bus)[0]
    return int(slack_gen_candidates[0]) if len(slack_gen_candidates) else 0


def _build_branch_flow_map(ppc_c: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the linear branch-flow map for one network state.

    Returns a map of the form
        flow = H @ inj_full + const

    Returns
    -------
    H : np.ndarray, shape (n_branch, nb)
        Linear sensitivity of branch flows to full bus injections.
    const : np.ndarray, shape (n_branch,)
        Constant term from phase shifters / transformer injections.
    rate_mw : np.ndarray, shape (n_branch,)
        Thermal limits RATE_A in MW.
    """
    bus = np.asarray(ppc_c["bus"], dtype=float)
    branch = np.asarray(ppc_c["branch"], dtype=float)

    Bbus, Bf, Pbusinj, Pfinj = _makeBdc_pp(bus, branch)

    ref_candidates = np.where(bus[:, BUS_TYPE] == REF)[0]
    ref = int(ref_candidates[0]) if len(ref_candidates) else 0
    keep = [i for i in range(bus.shape[0]) if i != ref]

    Bred = Bbus[np.ix_(keep, keep)]
    Bred_inv = np.linalg.inv(Bred)

    E = np.zeros((len(keep), bus.shape[0]), dtype=float)
    for r, i in enumerate(keep):
        E[r, i] = 1.0

    H = Bf[:, keep] @ Bred_inv @ E
    const = Pfinj - Bf[:, keep] @ Bred_inv @ Pbusinj[keep]
    rate_mw = np.asarray(branch[:, RATE_A], dtype=float).reshape(-1)

    return H, const, rate_mw


def _filter_valid_contingencies(
    ppc: Dict[str, np.ndarray],
    contingency_rows: Sequence[int],
) -> List[int]:
    """
    Filter candidate contingencies to only those usable in this DC pipeline.

    A contingency is kept only if:
    - the post-outage network remains connected
    - the DC flow map can be built without singular-matrix failure
    """
    valid: List[int] = []
    for row in contingency_rows:
        ppc_c = _apply_branch_outage(ppc, int(row))
        if not _is_ppc_connected(ppc_c):
            continue
        try:
            _build_branch_flow_map(ppc_c)
        except np.linalg.LinAlgError:
            continue
        valid.append(int(row))
    return valid


def build_dc_case(
    ppc: Dict[str, np.ndarray],
    contingency_rows: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Build the baseline-independent DC case dictionary.

    contains only network / physics / optimization data.
    does NOT assume any specific model prediction format.

    Parameters
    ----------
    ppc : dict
        PPC dictionary.
    contingency_rows : sequence of int, optional
        PPC branch indices to outage one-at-a-time.

    Returns
    -------
    dc_case : dict
        Shared DC backend data used for evaluation, optimization,
        feasibility checks, and optional projection/repair.
    """
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)

    nb = bus.shape[0]
    ng = gen.shape[0]
    base_mva = float(ppc["baseMVA"])

    ref_candidates = np.where(bus[:, BUS_TYPE] == REF)[0]
    slack_bus = int(ref_candidates[0]) if len(ref_candidates) else 0
    balance_idx = _default_balance_gen_index(ppc)

    pmin = np.asarray(gen[:, PMIN], dtype=float).reshape(-1)
    pmax = np.asarray(gen[:, PMAX], dtype=float).reshape(-1)
    gen_bus = np.round(gen[:, GEN_BUS]).astype(int).reshape(-1)
    load_buses = np.where(np.asarray(bus[:, PD], dtype=float) != 0.0)[0].astype(int)

    # Generator-to-bus incidence: bus_generation = Cg @ pg
    Cg = np.zeros((nb, ng), dtype=float)
    for j, b in enumerate(gen_bus):
        Cg[int(b), j] += 1.0

    a, b, c = extract_polynomial_cost(ppc, ng)

    H_list: List[np.ndarray] = []
    const_list: List[np.ndarray] = []
    rate_list: List[np.ndarray] = []

    Hnorm_list: List[np.ndarray] = []
    const_norm_list: List[np.ndarray] = []

    contingency_rows = _filter_valid_contingencies(ppc, list(contingency_rows or []))
    all_conts = [None] + list(contingency_rows)

    for outaged_row in all_conts:
        ppc_c = _apply_branch_outage(ppc, outaged_row)

        H_raw, const_raw, rate_mw = _build_branch_flow_map(ppc_c)

        rate_mw = np.where(np.isfinite(rate_mw) & (rate_mw > 0), rate_mw, 1e6)
        H_norm = H_raw / rate_mw[:, None]
        const_norm = const_raw / rate_mw

        H_list.append(H_raw)
        const_list.append(const_raw)
        rate_list.append(rate_mw)

        Hnorm_list.append(H_norm)
        const_norm_list.append(const_norm)

    return {
        "ppc": ppc,
        "baseMVA": base_mva,
        "nb": nb,
        "ng": ng,
        "slack_bus": slack_bus,
        "balance_idx": balance_idx,  # generic solver helper only
        "load_buses": load_buses,
        "gen_bus": gen_bus,
        "gen_signatures": build_gen_signatures(ppc),
        "pmin": pmin,
        "pmax": pmax,
        "pmin_pu": (pmin / base_mva).astype(np.float32),
        "pmax_pu": (pmax / base_mva).astype(np.float32),
        "a": a,
        "b": b,
        "c": c,
        "Cg": Cg,
        "H_list": H_list,
        "const_list": const_list,
        "rate_list": rate_list,
        "Hnorm_list": Hnorm_list,
        "const_norm_list": const_norm_list,
        "contingency_rows": list(contingency_rows or []),
        "base_pd": aggregate_bus_pd_from_ppc(ppc),
        "base_pg": np.asarray(gen[:, PG], dtype=float).reshape(-1),
    }


def compute_cost(dc_case: Dict[str, Any], pg: np.ndarray) -> float:
    """
    Compute quadratic generation cost for a dispatch vector `pg`.
    """
    pg = np.asarray(pg, dtype=float).reshape(-1)
    return float(np.sum(dc_case["a"] * pg * pg + dc_case["b"] * pg + dc_case["c"]))


def compute_flows(
    dc_case: Dict[str, Any],
    pg: np.ndarray,
    pd_bus: np.ndarray,
    which: int = 0,
) -> np.ndarray:
    """
    Compute branch flows for the base case or one contingency case.

    Parameters
    ----------
    which : int
        0 = base case
        1..K = contingency case index
    """
    H = dc_case["H_list"][which]
    const = dc_case["const_list"][which]
    Cg = dc_case["Cg"]
    inj = Cg @ np.asarray(pg, dtype=float).reshape(-1) - np.asarray(pd_bus, dtype=float).reshape(-1)
    return H @ inj + const


def feasibility_summary(
    dc_case: Dict[str, Any],
    pg: np.ndarray,
    pd_bus: np.ndarray,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Evaluate whether a generator dispatch `pg` is feasible for the DC case.

    Checks performed
    ----------------
    1) power balance:
         sum(pg) == sum(pd_bus)
    2) generator bounds:
         pmin <= pg <= pmax
    3) branch thermal limits:
         |flow| <= rate
       for base case and every retained contingency case
    """
    pg = np.asarray(pg, dtype=float).reshape(-1)
    pd_bus = np.asarray(pd_bus, dtype=float).reshape(-1)

    bal_err = abs(np.sum(pg) - np.sum(pd_bus))

    gen_upper = np.maximum(pg - dc_case["pmax"], 0.0)
    gen_lower = np.maximum(dc_case["pmin"] - pg, 0.0)
    gen_viol = float(np.max(np.r_[gen_upper, gen_lower])) if len(pg) else 0.0

    max_flow_viol = 0.0
    flow_viols: List[float] = []
    for i in range(len(dc_case["H_list"])):
        flows = compute_flows(dc_case, pg, pd_bus, which=i)
        rate = dc_case["rate_list"][i]
        viol = np.maximum(np.abs(flows) - rate, 0.0)
        this_max = float(np.max(viol)) if len(viol) else 0.0
        flow_viols.append(this_max)
        max_flow_viol = max(max_flow_viol, this_max)

    penalty = float(bal_err + gen_viol + max_flow_viol)
    feasible = (bal_err <= tol) and (gen_viol <= tol) and (max_flow_viol <= tol)

    return {
        "feasible": bool(feasible),
        "penalty": penalty,
        "balance_error": float(bal_err),
        "max_gen_violation": float(gen_viol),
        "max_flow_violation": float(max_flow_viol),
        "max_flow_violation_per_case": flow_viols,
    }


def reconstruct_theta(
    dc_case: Dict[str, Any],
    pg: np.ndarray,
    pd_bus: np.ndarray,
    which: int = 0,
) -> np.ndarray:
    """
    Reconstruct DC bus voltage angles for one case.

    Parameters
    ----------
    which : int
        0 means base case
        1..K means contingency case index relative to dc_case["contingency_rows"]

    Notes
    -----
    Mainly useful for debugging. Feasibility checks use the precomputed
    flow maps directly.
    """
    ppc_c = _apply_branch_outage(
        dc_case["ppc"],
        None if which == 0 else dc_case["contingency_rows"][which - 1],
    )
    bus = np.asarray(ppc_c["bus"], dtype=float)
    branch = np.asarray(ppc_c["branch"], dtype=float)
    Bbus, _, Pbusinj, _ = _makeBdc_pp(bus, branch)

    ref_candidates = np.where(bus[:, BUS_TYPE] == REF)[0]
    ref = int(ref_candidates[0]) if len(ref_candidates) else 0
    keep = [i for i in range(bus.shape[0]) if i != ref]

    inj = dc_case["Cg"] @ np.asarray(pg, dtype=float).reshape(-1) - np.asarray(pd_bus, dtype=float).reshape(-1)
    theta_red = np.linalg.solve(Bbus[np.ix_(keep, keep)], inj[keep] - Pbusinj[keep])

    theta = np.zeros(bus.shape[0], dtype=float)
    theta[keep] = theta_red
    theta[ref] = 0.0
    return theta


def _linear_flow_constraints(
    dc_case: Dict[str, Any],
    pd_bus: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert branch thermal limits into stacked linear constraints on pg.

    Since
        flow = H @ (Cg @ pg - pd_bus) + const
    we can write
        flow = M @ pg + d

    and enforce
        -rate <= M @ pg + d <= rate

    Returns
    -------
    A, lb, ub such that
        lb <= A @ pg <= ub
    """
    pd_bus = np.asarray(pd_bus, dtype=float).reshape(-1)

    mats = []
    lbs = []
    ubs = []
    Cg = dc_case["Cg"]

    for H, const, rate in zip(dc_case["H_list"], dc_case["const_list"], dc_case["rate_list"]):
        M = H @ Cg
        d = const - H @ pd_bus
        mats.append(M)
        lbs.append(-rate - d)
        ubs.append(rate - d)

    A = np.vstack(mats) if mats else np.zeros((0, dc_case["ng"]), dtype=float)
    lb = np.concatenate(lbs) if lbs else np.zeros((0,), dtype=float)
    ub = np.concatenate(ubs) if ubs else np.zeros((0,), dtype=float)
    return A, lb, ub


def solve_sc_dcopf(
    dc_case: Dict[str, Any],
    pd_bus: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Solve the security-constrained DC-OPF for a given load vector.

    Decision variable
    -----------------
    pg : generator dispatch in MW

    Objective
    ---------
    minimize quadratic generation cost

    Constraints
    -----------
    1) total generation equals total load
    2) generator bounds
    3) branch thermal limits for base case + contingencies
    """
    pd_bus = np.asarray(pd_bus, dtype=float).reshape(-1)
    ng = int(dc_case["ng"])
    target_load = float(np.sum(pd_bus))

    Aflow, lflow, uflow = _linear_flow_constraints(dc_case, pd_bus)

    lin_cons = [
        LinearConstraint(
            np.ones((1, ng), dtype=float),
            lb=np.array([target_load]),
            ub=np.array([target_load]),
        )
    ]

    if Aflow.shape[0] > 0:
        lin_cons.append(LinearConstraint(Aflow, lb=lflow, ub=uflow))

    bounds = Bounds(dc_case["pmin"], dc_case["pmax"])
    a = np.asarray(dc_case["a"], dtype=float)
    b = np.asarray(dc_case["b"], dtype=float)

    if x0 is None:
        x0 = np.asarray(dc_case["base_pg"], dtype=float).reshape(-1)

    x0 = np.clip(np.asarray(x0, dtype=float).reshape(-1), dc_case["pmin"], dc_case["pmax"])

    # Use one default balancing generator only as an initialization helper.
    mismatch = target_load - float(np.sum(x0))
    balance_idx = int(dc_case.get("balance_idx", 0))
    if 0 <= balance_idx < len(x0):
        x0[balance_idx] += mismatch
    x0 = np.clip(x0, dc_case["pmin"], dc_case["pmax"])

    def obj(pg: np.ndarray) -> float:
        return float(np.sum(a * pg * pg + b * pg))

    def grad(pg: np.ndarray) -> np.ndarray:
        return 2.0 * a * pg + b

    def hess(_pg: np.ndarray) -> np.ndarray:
        return np.diag(2.0 * a + 1e-9)

    res = minimize(
        obj,
        x0,
        method="trust-constr",
        jac=grad,
        hess=hess,
        constraints=lin_cons,
        bounds=bounds,
        options={"maxiter": 500, "verbose": 0},
    )

    pg = np.asarray(res.x if res.success else x0, dtype=float).reshape(-1)

    flows_ok = True
    for i in range(len(dc_case["H_list"])):
        flows = compute_flows(dc_case, pg, pd_bus, which=i)
        rate = dc_case["rate_list"][i]
        if np.any(np.abs(flows) - rate > 1e-6):
            flows_ok = False
            break

    feasible = (
        abs(np.sum(pg) - np.sum(pd_bus)) <= 1e-6
        and np.all(pg >= dc_case["pmin"] - 1e-6)
        and np.all(pg <= dc_case["pmax"] + 1e-6)
        and flows_ok
    )

    return {
        "success": bool(res.success) and bool(feasible),
        "status": getattr(res, "status", None),
        "message": getattr(res, "message", ""),
        "pg": pg,
        "cost": compute_cost(dc_case, pg),
        "nit": int(getattr(res, "nit", 0) or 0),
    }


def project_pg_to_feasible(
    dc_case: Dict[str, Any],
    pd_bus: np.ndarray,
    pg_hat: np.ndarray,
) -> Dict[str, Any]:
    """
    Project an arbitrary dispatch `pg_hat` onto the feasible region.

    The projection minimizes L1 distance to the candidate dispatch while
    enforcing:
    - total power balance
    - generator bounds
    - branch thermal limits for base case + contingencies
    """
    pd_bus = np.asarray(pd_bus, dtype=float).reshape(-1)
    pg_hat = np.asarray(pg_hat, dtype=float).reshape(-1)
    ng = int(dc_case["ng"])

    Aflow, lflow, uflow = _linear_flow_constraints(dc_case, pd_bus)

    c = np.concatenate([np.zeros(ng, dtype=float), np.ones(ng, dtype=float)])

    Aub = []
    bub = []

    # pg - t <= pg_hat
    M = np.zeros((ng, 2 * ng), dtype=float)
    M[np.arange(ng), np.arange(ng)] = 1.0
    M[np.arange(ng), ng + np.arange(ng)] = -1.0
    Aub.append(M)
    bub.append(pg_hat)

    # -pg - t <= -pg_hat
    M = np.zeros((ng, 2 * ng), dtype=float)
    M[np.arange(ng), np.arange(ng)] = -1.0
    M[np.arange(ng), ng + np.arange(ng)] = -1.0
    Aub.append(M)
    bub.append(-pg_hat)

    if Aflow.shape[0] > 0:
        M = np.zeros((Aflow.shape[0], 2 * ng), dtype=float)
        M[:, :ng] = Aflow
        Aub.append(M)
        bub.append(uflow)

        M = np.zeros((Aflow.shape[0], 2 * ng), dtype=float)
        M[:, :ng] = -Aflow
        Aub.append(M)
        bub.append(-lflow)

    Aeq = np.zeros((1, 2 * ng), dtype=float)
    Aeq[0, :ng] = 1.0
    beq = np.array([float(np.sum(pd_bus))], dtype=float)

    lb = np.concatenate([dc_case["pmin"], np.zeros(ng, dtype=float)])
    ub = np.concatenate([dc_case["pmax"], np.full(ng, np.inf, dtype=float)])

    res = linprog(
        c=c,
        A_ub=np.vstack(Aub) if Aub else None,
        b_ub=np.concatenate(bub) if bub else None,
        A_eq=Aeq,
        b_eq=beq,
        bounds=list(zip(lb, ub)),
        method="highs",
    )

    pg = np.asarray(res.x[:ng], dtype=float) if res.success else np.asarray(pg_hat, dtype=float)

    return {
        "success": bool(res.success),
        "message": getattr(res, "message", ""),
        "pg": pg,
        "cost": compute_cost(dc_case, pg),
    }