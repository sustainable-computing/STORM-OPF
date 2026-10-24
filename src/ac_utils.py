'''
this file has common util functions required for different ac OPF baselines. 
'''
import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandapower as pp
import pandas as pd

import src.global_vars as gvars


def prep_constraints(net):
    """Fill missing network limits."""
    if "min_vm_pu" not in net.bus:
        net.bus["min_vm_pu"] = 0.95
    if "max_vm_pu" not in net.bus:
        net.bus["max_vm_pu"] = 1.05

    for c in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if c not in net.gen:
            net.gen[c] = np.nan

    net.gen["min_p_mw"] = net.gen["min_p_mw"].fillna(0.0)
    net.gen["max_p_mw"] = net.gen["max_p_mw"].fillna(np.maximum(1.0, net.gen["p_mw"] * 1.2))
    net.gen["min_q_mvar"] = net.gen["min_q_mvar"].fillna(-1e3)
    net.gen["max_q_mvar"] = net.gen["max_q_mvar"].fillna(+1e3)

    if len(net.line):
        if "max_loading_percent" not in net.line:
            net.line["max_loading_percent"] = 100.0
        net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(100.0)
    if len(net.trafo):
        if "max_loading_percent" not in net.trafo:
            net.trafo["max_loading_percent"] = 100.0
        net.trafo["max_loading_percent"] = net.trafo["max_loading_percent"].fillna(100.0)
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        if "max_loading_percent" not in net.trafo3w:
            net.trafo3w["max_loading_percent"] = 100.0
        net.trafo3w["max_loading_percent"] = net.trafo3w["max_loading_percent"].fillna(100.0)


def aggregate_bus_vector(net, series_name: str) -> np.ndarray:
    """Return summ of load value per bus."""
    n_b = len(net.bus)
    vec = np.zeros(n_b, dtype=float)
    if len(net.load):
        for _, row in net.load.iterrows():
            vec[int(row["bus"])] += float(row[series_name])
    return vec


def build_load_map(net) -> Dict[int, np.ndarray]:
    """Map bus index to load rows."""
    out: Dict[int, np.ndarray] = {}
    if len(net.load):
        buses = net.load["bus"].to_numpy()
        idxs = net.load.index.to_numpy()
        for b in net.bus.index.to_numpy():
            mask = buses == b
            if mask.any():
                out[int(b)] = idxs[mask]
    return out


def apply_loads_to_net(net, P_D: np.ndarray, Q_D: np.ndarray, load_map: Optional[Dict[int, np.ndarray]] = None):
    """Write bus demand to load rows."""
    if load_map is None:
        load_map = build_load_map(net)
    if len(net.load):
        for bus in net.bus.index:
            idxs = load_map.get(int(bus))
            if idxs is None or len(idxs) == 0:
                continue
            share_p = float(P_D[int(bus)]) / len(idxs)
            share_q = float(Q_D[int(bus)]) / len(idxs)
            net.load.loc[idxs, "p_mw"] = share_p
            net.load.loc[idxs, "q_mvar"] = share_q


def run_pf(net, vm_init=None, va_init=None, enforce_q_lims=False):
    """Run AC power flow."""
    kwargs = dict(enforce_q_lims=bool(enforce_q_lims), calculate_voltage_angles=True)
    if vm_init is not None or va_init is not None:
        pp.runpp(net, init="auto", init_vm_pu=vm_init, init_va_degree=va_init, **kwargs)
    else:
        pp.runpp(net, **kwargs)


def _parse_jsonish_list(value: Optional[Sequence[Any]]):
    # Normalize input into a Python list. return the list
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [s]
    return list(value)


def _all_in_service_contingencies(net) -> List[Dict[str, int]]:
    # Collect all in-service outage candidates from lines and transformers.
    
    out: List[Dict[str, int]] = []
    if hasattr(net, "line") and len(net.line):
        for idx, row in net.line.iterrows():
            if bool(row.get("in_service", True)):
                out.append({"kind": "line", "index": int(idx)})
    if hasattr(net, "trafo") and len(net.trafo):
        for idx, row in net.trafo.iterrows():
            if bool(row.get("in_service", True)):
                out.append({"kind": "trafo", "index": int(idx)})
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        for idx, row in net.trafo3w.iterrows():
            if bool(row.get("in_service", True)):
                out.append({"kind": "trafo3w", "index": int(idx)})
    return out


def _resolve_one_item(net, item: Any) -> Optional[Dict[str, int]]:
    # Convert one 'contingency entry' into a {"kind", "index"} entry.
    # this function is used to convert cintingencies into a consistent format
    
    if isinstance(item, dict):
        kind = item.get("kind") or item.get("type") or item.get("table")
        index = item.get("index")
        if index is None:
            index = item.get("row")
        if kind is None or index is None:
            return None
        kind = str(kind).strip().lower()
        index = int(index)
        if kind in {"line", "trafo", "trafo3w"}:
            table = getattr(net, kind, None)
            if table is not None and index in table.index:
                return {"kind": kind, "index": index}
        return None
    if isinstance(item, int):
        # Integer input is treated as a line index
        if hasattr(net, "line") and item in net.line.index:
            return {"kind": "line", "index": int(item)}
        return None
    if isinstance(item, str):
        # Accept plain indices like "3" or typed references like "line:3"
        s = item.strip().lower()
        if not s:
            return None
        if s.isdigit():
            idx = int(s)
            if hasattr(net, "line") and idx in net.line.index:
                return {"kind": "line", "index": idx}
            return None
        if ":" in s:
            kind, idx_s = s.split(":", 1)
            kind = kind.strip()
            idx_s = idx_s.strip()
            if not idx_s.isdigit():
                return None
            idx = int(idx_s)
            if kind in {"line", "trafo", "trafo3w"}:
                table = getattr(net, kind, None)
                if table is not None and idx in table.index:
                    return {"kind": kind, "index": idx}
    return None


def resolve_contingencies(net, contingency_spec=None, scenario_path: Optional[str] = None) -> List[Dict[str, int]]:
    """Build contingency list."""
    spec = _parse_jsonish_list(contingency_spec)
    if spec is None and scenario_path is not None:
        sidecar = scenario_path + ".contingencies.json"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                spec = raw.get("contingencies")
                if spec is None:
                    spec = raw.get("elements")
            elif isinstance(raw, list):
                spec = raw
    if spec is None:
        return _all_in_service_contingencies(net)
    if len(spec) == 0:
        return []

    out: List[Dict[str, int]] = []
    for item in spec:
        resolved = _resolve_one_item(net, item)
        if resolved is not None:
            out.append(resolved)
    uniq = {(c["kind"], c["index"]): c for c in out}
    return [uniq[k] for k in sorted(uniq.keys())]


def contingency_name(cont: Dict[str, int]) -> str:
    return f"{cont['kind']}:{int(cont['index'])}"


def apply_contingency(net, cont: Dict[str, int]):
    """Outage one line or transformer."""
    kind = cont["kind"]
    index = int(cont["index"])
    getattr(net, kind).at[index, "in_service"] = False
    return net


def _active_bus_list(net) -> List[int]:
    if "in_service" in net.bus.columns:
        return [int(i) for i, flag in net.bus["in_service"].items() if bool(flag)]
    return [int(i) for i in net.bus.index]


def _build_active_bus_graph(net) -> Dict[int, set]:
    active = set(_active_bus_list(net))
    adj: Dict[int, set] = {b: set() for b in active}
    if hasattr(net, "line") and len(net.line):
        for _, row in net.line.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            fb = int(row["from_bus"])
            tb = int(row["to_bus"])
            if fb in active and tb in active:
                adj[fb].add(tb)
                adj[tb].add(fb)
    if hasattr(net, "trafo") and len(net.trafo):
        for _, row in net.trafo.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            hv = int(row["hv_bus"])
            lv = int(row["lv_bus"])
            if hv in active and lv in active:
                adj[hv].add(lv)
                adj[lv].add(hv)
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        for _, row in net.trafo3w.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            buses = [int(row["hv_bus"]), int(row["mv_bus"]), int(row["lv_bus"])]
            buses = [b for b in buses if b in active]
            for i in range(len(buses)):
                for j in range(i + 1, len(buses)):
                    adj[buses[i]].add(buses[j])
                    adj[buses[j]].add(buses[i])
    return adj


def network_is_connected(net) -> bool:
    """Return True when active buses remain connected."""
    active = _active_bus_list(net)
    if len(active) <= 1:
        return True
    adj = _build_active_bus_graph(net)
    slack_bus = None
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        for _, row in net.ext_grid.iterrows():
            if bool(row.get("in_service", True)):
                candidate = int(row["bus"])
                if candidate in adj:
                    slack_bus = candidate
                    break
    start = slack_bus if slack_bus is not None else active[0]
    seen = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(v for v in adj.get(u, ()) if v not in seen)
    return len(seen) == len(active)


def filter_valid_contingencies(base_net, contingencies: Sequence[Dict[str, int]]) -> List[Dict[str, int]]:
    """Remove islanding contingencies."""
    valid: List[Dict[str, int]] = []
    for cont in contingencies:
        net_c = copy.deepcopy(base_net)
        apply_contingency(net_c, cont)
        if network_is_connected(net_c):
            valid.append({"kind": cont["kind"], "index": int(cont["index"])})
    return valid


def compute_penalty_from_net(net, ctol=gvars.ctol) -> Tuple[float, Dict[str, float]]:
    """Return scalar penalty and violation summary."""
    pen = 0.0
    max_term = 0.0
    baseMVA = float(net.sn_mva)

    if len(net.gen):
        g = net.gen
        rg = net.res_gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool)

        if {"min_p_mw", "max_p_mw"}.issubset(g.columns):
            p = rg["p_mw"].to_numpy()[on]
            pmin = g["min_p_mw"].to_numpy()[on]
            pmax = g["max_p_mw"].to_numpy()[on]
            u = np.maximum(p - pmax, 0.0) / baseMVA
            l = np.maximum(pmin - p, 0.0) / baseMVA
            u[u < ctol] = 0.0
            l[l < ctol] = 0.0
            pen += float(u.sum() + l.sum())
            max_term = max(max_term, float(np.max(u)) if u.size else 0.0, float(np.max(l)) if l.size else 0.0)

        if {"min_q_mvar", "max_q_mvar"}.issubset(g.columns):
            q = rg["q_mvar"].to_numpy()[on]
            qmin = g["min_q_mvar"].to_numpy()[on]
            qmax = g["max_q_mvar"].to_numpy()[on]
            u = np.maximum(q - qmax, 0.0) / baseMVA
            l = np.maximum(qmin - q, 0.0) / baseMVA
            u[u < ctol] = 0.0
            l[l < ctol] = 0.0
            pen += float(u.sum() + l.sum())
            max_term = max(max_term, float(np.max(u)) if u.size else 0.0, float(np.max(l)) if l.size else 0.0)

    if len(net.bus):
        v = net.res_bus["vm_pu"].to_numpy()
        v_lo = net.bus.get("min_vm_pu", 0.95)
        v_hi = net.bus.get("max_vm_pu", 1.05)
        v_lo = v_lo if np.isscalar(v_lo) else v_lo.to_numpy()
        v_hi = v_hi if np.isscalar(v_hi) else v_hi.to_numpy()
        u = np.maximum(v - v_hi, 0.0)
        l = np.maximum(v_lo - v, 0.0)
        u[u < ctol] = 0.0
        l[l < ctol] = 0.0
        pen += float(u.sum() + l.sum())
        max_term = max(max_term, float(np.max(u)) if u.size else 0.0, float(np.max(l)) if l.size else 0.0)

    if len(net.line):
        allow = net.line.get("max_loading_percent", 100.0)
        allow = allow if np.isscalar(allow) else allow.to_numpy()
        load = net.res_line["loading_percent"].to_numpy()
        over = np.maximum(load - allow, 0.0) / 100.0
        over[over < ctol] = 0.0
        pen += float(over.sum())
        max_term = max(max_term, float(np.max(over)) if over.size else 0.0)

    if len(net.trafo):
        allow = net.trafo.get("max_loading_percent", 100.0)
        allow = allow if np.isscalar(allow) else allow.to_numpy()
        load = net.res_trafo["loading_percent"].to_numpy()
        over = np.maximum(load - allow, 0.0) / 100.0
        over[over < ctol] = 0.0
        pen += float(over.sum())
        max_term = max(max_term, float(np.max(over)) if over.size else 0.0)

    if hasattr(net, "trafo3w") and len(net.trafo3w):
        allow = net.trafo3w.get("max_loading_percent", 100.0)
        allow = allow if np.isscalar(allow) else allow.to_numpy()
        load = net.res_trafo3w["loading_percent"].to_numpy()
        over = np.maximum(load - allow, 0.0) / 100.0
        over[over < ctol] = 0.0
        pen += float(over.sum())
        max_term = max(max_term, float(np.max(over)) if over.size else 0.0)

    return float(pen), {"max_violation": float(max_term)}


def compute_operating_cost(net) -> float:
    """Compute generator and slack cost when available."""
    total = 0.0
    if hasattr(net, "poly_cost") and len(net.poly_cost):
        for _, row in net.poly_cost.iterrows():
            et = str(row["et"])
            idx = int(row["element"])
            p = 0.0
            if et == "gen" and len(net.res_gen) and idx in net.res_gen.index:
                p = float(net.res_gen.at[idx, "p_mw"])
            elif et == "ext_grid" and len(net.res_ext_grid) and idx in net.res_ext_grid.index:
                p = float(net.res_ext_grid.at[idx, "p_mw"])
            else:
                continue
            cp2 = float(row.get("cp2_eur_per_mw2", 0.0) or 0.0)
            cp1 = float(row.get("cp1_eur_per_mw", 0.0) or 0.0)
            cp0 = float(row.get("cp0_eur", 0.0) or 0.0)
            total += cp2 * p * p + cp1 * p + cp0
        return float(total)

    if len(net.res_gen):
        p = net.res_gen["p_mw"].to_numpy(dtype=float)
        return float(1e-3 * np.sum(p * p))
    return 0.0


def get_orig_pq_bus_idx(net):
    """Return buses that are not ext_grid buses and not gen buses."""
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
    vm_tol: float = 1e-3,
    pct_tol: float = 1e-3,
    pq_tol: float = 1e-6,
    require_converged: bool = True,
    vm_bus_idx=None,
) -> bool:
    """Return True iff all inequality constraints are satisfied within tolerances."""
    if require_converged:
        pf_conv = bool(getattr(net, "converged", False))
        opf_conv = bool(net.get("OPF_converged", False))
        if not (pf_conv or opf_conv):
            return False

    if len(net.bus):
        bus_idx = net.bus.index
        vm = net.res_bus.loc[bus_idx, "vm_pu"].to_numpy()
        v_lo_col = net.bus.get("min_vm_pu", 0.95)
        v_hi_col = net.bus.get("max_vm_pu", 1.05)
        v_lo = v_lo_col if np.isscalar(v_lo_col) else v_lo_col.to_numpy()
        v_hi = v_hi_col if np.isscalar(v_hi_col) else v_hi_col.to_numpy()
        if vm_bus_idx is not None:
            mask = bus_idx.isin(np.asarray(vm_bus_idx))
            vm = vm[mask]
            v_lo = v_lo[mask]
            v_hi = v_hi[mask]
        viol_vm_lo = (vm + vm_tol) < v_lo
        viol_vm_hi = (vm - vm_tol) > v_hi
        if np.any(viol_vm_lo) or np.any(viol_vm_hi):
            return False

    pq_rel = 1e-3
    if len(net.gen):
        g = net.gen
        rg = net.res_gen
        on = g["in_service"].fillna(True).to_numpy().astype(bool)
        if {"min_p_mw", "max_p_mw"}.issubset(g.columns):
            p = rg.loc[on, "p_mw"].to_numpy()
            pmin = g.loc[on, "min_p_mw"].to_numpy()
            pmax = g.loc[on, "max_p_mw"].to_numpy()
            has_pmin = ~np.isnan(pmin)
            has_pmax = ~np.isnan(pmax)
            if np.any(has_pmax):
                p_u = p[has_pmax]
                pmax_u = pmax[has_pmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(pmax_u), 1.0), pq_tol)
                if np.any((p_u - pmax_u) > tol_u):
                    return False
            if np.any(has_pmin):
                p_l = p[has_pmin]
                pmin_l = pmin[has_pmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(pmin_l), 1.0), pq_tol)
                if np.any((pmin_l - p_l) > tol_l):
                    return False
        if {"min_q_mvar", "max_q_mvar"}.issubset(g.columns):
            q = rg.loc[on, "q_mvar"].to_numpy()
            qmin = g.loc[on, "min_q_mvar"].to_numpy()
            qmax = g.loc[on, "max_q_mvar"].to_numpy()
            has_qmin = ~np.isnan(qmin)
            has_qmax = ~np.isnan(qmax)
            if np.any(has_qmax):
                q_u = q[has_qmax]
                qmax_u = qmax[has_qmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(qmax_u), 1.0), pq_tol)
                if np.any((q_u - qmax_u) > tol_u):
                    return False
            if np.any(has_qmin):
                q_l = q[has_qmin]
                qmin_l = qmin[has_qmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(qmin_l), 1.0), pq_tol)
                if np.any((qmin_l - q_l) > tol_l):
                    return False

    if len(net.sgen) and hasattr(net, "res_sgen") and len(net.res_sgen):
        s = net.sgen
        rs = net.res_sgen
        on = s["in_service"].fillna(True).to_numpy().astype(bool)
        if {"min_p_mw", "max_p_mw"}.issubset(s.columns):
            p = rs.loc[on, "p_mw"].to_numpy()
            pmin = s.loc[on, "min_p_mw"].to_numpy()
            pmax = s.loc[on, "max_p_mw"].to_numpy()
            has_pmin = ~np.isnan(pmin)
            has_pmax = ~np.isnan(pmax)
            if np.any(has_pmax):
                p_u = p[has_pmax]
                pmax_u = pmax[has_pmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(pmax_u), 1.0), pq_tol)
                if np.any((p_u - pmax_u) > tol_u):
                    return False
            if np.any(has_pmin):
                p_l = p[has_pmin]
                pmin_l = pmin[has_pmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(pmin_l), 1.0), pq_tol)
                if np.any((pmin_l - p_l) > tol_l):
                    return False
        if {"min_q_mvar", "max_q_mvar"}.issubset(s.columns):
            q = rs.loc[on, "q_mvar"].to_numpy()
            qmin = s.loc[on, "min_q_mvar"].to_numpy()
            qmax = s.loc[on, "max_q_mvar"].to_numpy()
            has_qmin = ~np.isnan(qmin)
            has_qmax = ~np.isnan(qmax)
            if np.any(has_qmax):
                q_u = q[has_qmax]
                qmax_u = qmax[has_qmax]
                tol_u = np.maximum(pq_rel * np.maximum(np.abs(qmax_u), 1.0), pq_tol)
                if np.any((q_u - qmax_u) > tol_u):
                    return False
            if np.any(has_qmin):
                q_l = q[has_qmin]
                qmin_l = qmin[has_qmin]
                tol_l = np.maximum(pq_rel * np.maximum(np.abs(qmin_l), 1.0), pq_tol)
                if np.any((qmin_l - q_l) > tol_l):
                    return False

    def _violates_loading(res_tbl, elem_tbl, on_mask, pct_tol):
        limit = (elem_tbl.get("max_loading_percent", 100.0).fillna(100.0).to_numpy())[on_mask]
        lpct = res_tbl.loc[on_mask, "loading_percent"].to_numpy()
        ok_mask = ~np.isnan(lpct) & ~np.isnan(limit)
        return np.any(lpct[ok_mask] > (limit[ok_mask] + pct_tol))

    if len(net.line) and hasattr(net, "res_line") and len(net.res_line):
        on = net.line["in_service"].fillna(True).to_numpy().astype(bool)
        if _violates_loading(net.res_line, net.line, on, pct_tol):
            return False
    if len(net.trafo) and hasattr(net, "res_trafo") and len(net.res_trafo):
        on = net.trafo["in_service"].fillna(True).to_numpy().astype(bool)
        if _violates_loading(net.res_trafo, net.trafo, on, pct_tol):
            return False
    if hasattr(net, "trafo3w") and len(net.trafo3w) and hasattr(net, "res_trafo3w") and len(net.res_trafo3w):
        on = net.trafo3w["in_service"].fillna(True).to_numpy().astype(bool)
        if _violates_loading(net.res_trafo3w, net.trafo3w, on, pct_tol):
            return False
    return True
