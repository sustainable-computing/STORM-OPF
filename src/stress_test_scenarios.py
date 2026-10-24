import numpy as np
import copy
import random 

# ---------- Scenario registry ----------
SCENARIOS = {}  # name -> callable

def register_scenario(name: str):
    """Decorator to register a scenario factory under a string name."""
    def deco(fn):
        if name in SCENARIOS:
            raise KeyError(f"Scenario '{name}' already registered")
        SCENARIOS[name] = fn
        return fn
    return deco


# ---------- helpers ----------
def _rng(seed=None):
    return np.random.default_rng(seed)

def _pick_indices(n, k, rng):
    k = min(k, n)
    return rng.choice(np.arange(n), size=k, replace=False) if n and k else np.array([], dtype=int)

def _scale(df, col, factor):
        if df is not None and len(df) and col in df.columns:
            df[col] = df[col] * factor

# ---------- Load level extremes ----------
def scenario_load_level_extremes(net, severity=1.0, seed=None, mode="high"):
    """
    mode = 'high' (near limits) or 'low' (light-load)
    severity: higher => more extreme
    Each load gets its own random scaling factor.
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)

    n_loads = len(net_scn.load)
    if n_loads == 0:
        return net_scn, {"category": "Input Distribution Stressors",
                         "scenario": f"Load level extremes ({mode})",
                         "note": "no loads"}

    if mode == "high":
        # push loads up: base around 1 + 0.5*severity, but with random spread
        base = 1.0 + 0.5 * severity
        factors = rng.normal(loc=base, scale=0.1 * severity, size=n_loads)
        factors = np.clip(factors, 1.0, None)   # don’t reduce loads in "high"
    else:  # "low"
        #k = np.clip(1.0 - 0.5 * severity, 0.40, 1.0) #floor at 40%
        #factors = np.full(n_loads, k)
        base = 1.0 - 0.5 * severity
        sigma = min(0.1 * severity, 0.05)       
        factors = rng.normal(loc=base, scale=sigma, size=n_loads)
        factors = np.clip(factors, 0.40, 0.95)  #to prevent lots of non-converging scenarios

    # apply individual factors
    net_scn.load["p_mw"] = net_scn.load["p_mw"] * factors
    if "q_mvar" in net_scn.load:
        net_scn.load["q_mvar"] = net_scn.load["q_mvar"] * factors

    meta = {
        "category": "Input Distribution Stressors",
        "scenario": f"Load level extremes ({mode})",
        "factors": factors.tolist()
    }
    return net_scn, meta

# ---------- Renewable generation variability ----------
def scenario_renewable_variability(net, severity=1.0, seed=None):
    """
    Scale PV/Wind sgens up/down together.
    Note: not all the test cases has renewables.
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)

    if "sgen" in net_scn and len(net_scn.sgen):
        # Correlated multiplier: around 1.0, widened by severity
        base = 1.0
        spread = 0.8 * severity   # up to ~+=80% at severity=1
        factor = max(0.0, base + rng.normal(0.0, 0.3*severity))
        factor = np.clip(factor, 0.0, 1.0 + spread)

        #identify RES by 'type' if available; else apply to all sgens
        mask = np.ones(len(net_scn.sgen), dtype=bool)
        if "type" in net_scn.sgen.columns:
            mask = net_scn.sgen["type"].str.lower().fillna("").str.contains("pv|solar|wind")

        idx = net_scn.sgen.index[mask]
        net_scn.sgen.loc[idx, "p_mw"] *= factor
        # Usually PV models run ~unity PF i.e. keep the power factor constant 
        # by changing q_mvar to the same ratio
        if "q_mvar" in net_scn.sgen.columns:
            net_scn.sgen.loc[idx, "q_mvar"] *= factor

        meta = {"category": "Input Distribution Stressors",
                "scenario": "Renewable generation variability",
                "factor": factor, "affected_sgens": idx.tolist()}
    else:
        print('This test case has no sgen; terminating this case.')
        exit(0)
    return net_scn, meta

# ---------- Fuel price shocks ----------
def scenario_fuel_price_shocks(net, severity=1.0, seed=None, shock="up"):
    """
    Shock generator marginal costs (cp1, cp2).
    shock = 'up' (price spike) or 'down' (price crash).
    """
    net_scn = copy.deepcopy(net)

    if "poly_cost" not in net_scn or len(net_scn.poly_cost) == 0:
        raise RuntimeError("No poly_cost found. Define costs before applying price shocks.")

    # Global multiplicative shock; severity controls magnitude
    base = 1.0
    delta = 0.8 * severity    # +-80% at severity=1
    mult = base + delta if shock == "up" else max(0.05, base - delta)

    # Apply to generator costs
    pc = net_scn.poly_cost
    
    mask = pc["et"].isin(["gen", "ext_grid"])  
    for col in pc.columns:
        if col.lower() in ("cp1_eur_per_mw", "cp2_eur_per_mw2"):
            pc.loc[mask, col] *= mult

    meta = {"category": "Input Distribution Stressors",
            "scenario": f"Fuel price shocks ({shock})",
            "multiplier": mult}
    return net_scn, meta

# ---------- Geographic load/generation shifts ----------
def scenario_geographic_shift(net, severity=1.0, seed=None, fraction=0.2, pct= 0.25):
    """
    Move a fraction of total load from random donor buses to random receiver buses.
    Preserves total system demand.
    'fraction' = share of total load to relocate (0..1).
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)

    if not len(net_scn.load):
        return net_scn, {"category": "Input Distribution Stressors",
                         "scenario": "Geographic load/generation shifts",
                         "note": "no loads to move"}

    total_p = net_scn.load["p_mw"].sum()
    total_q = net_scn.load["q_mvar"].sum() if "q_mvar" in net_scn.load else 0.0

    move_p = fraction * total_p * severity
    move_q = fraction * total_q * severity

    n = len(net_scn.load)

    k_each = max(1, int(pct * n))          # donors and receivers size
    k_total = min(2 * k_each, n)            # total picked (<= n)

    picked = _pick_indices(n, k_total, rng) # unique indices
    rng.shuffle(picked)

    donors = picked[:k_each]
    receivers = picked[k_each:2*k_each]

    # Compute donor contributions proportional to their sizes
    donor_p = net_scn.load.iloc[donors]["p_mw"].to_numpy()
    donor_q = net_scn.load.iloc[donors]["q_mvar"].to_numpy() if "q_mvar" in net_scn.load else np.zeros_like(donor_p)
    donor_p_share = donor_p / (donor_p.sum() + 1e-9)
    donor_q_share = donor_q / (abs(donor_q).sum() + 1e-9) if donor_q.size else np.zeros_like(donor_p)

    # Remove from donors
    net_scn.load.iloc[donors, net_scn.load.columns.get_loc("p_mw")] = np.maximum(
        0.0, donor_p - donor_p_share * move_p
    )
    if "q_mvar" in net_scn.load:
        net_scn.load.iloc[donors, net_scn.load.columns.get_loc("q_mvar")] = donor_q - donor_q_share * move_q

    # Distribute to receivers proportional to current size (or uniform if zero)
    recv_p = net_scn.load.iloc[receivers]["p_mw"].to_numpy()
    weights = recv_p
    if weights.sum() <= 1e-9:
        weights = np.ones_like(recv_p)
    weights = weights / weights.sum()

    net_scn.load.iloc[receivers, net_scn.load.columns.get_loc("p_mw")] = recv_p + weights * move_p
    if "q_mvar" in net_scn.load:
        recv_q = net_scn.load.iloc[receivers]["q_mvar"].to_numpy()
        if abs(recv_q).sum() <= 1e-9:
            wq = weights
        else:
            wq = np.abs(recv_q) / np.abs(recv_q).sum()
        net_scn.load.iloc[receivers, net_scn.load.columns.get_loc("q_mvar")] = recv_q + wq * move_q

    meta = {"category": "Input Distribution Stressors",
            "scenario": "Geographic load/generation shifts",
            "moved_p_mw": float(move_p),
            "moved_q_mvar": float(move_q),
            "donors": net_scn.load.index[donors].tolist(),
            "receivers": net_scn.load.index[receivers].tolist()}
    return net_scn, meta


def scenario_extreme_temperature(net, delta_c, seed=None):
    """
    Following paper: impact of cold waves & heat waves in energy production sectors
    Apply extreme temperature effects with minimal knobs (only delta_c in degree Cel.)
    Implements:
      1) Generation availability: -2% per deg. Cel.  (gens & sgens)
      2) Operating costs: staff +50..100% and stock +10..20%  (poly_cost cp1/cp2)
      3) System losses:
           - transformer capacity: -1% per deg. Cel. (max_loading_percent)
           - copper resistance: +0.4% per deg. Cel (line r_ohm_per_km)
           - loss adder: +1% per 3 deg. Cel. applied to demand
      4) High consumption: demand inc. linearly with delta_T, capped at +21%
    Returns: (net_scn, meta)
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)
    dT = float(delta_c)
    meta = {"scenario": "extreme_temperature", "delta_c": dT}

    # -------------------------------
    # 1) Less generation: -2%/ deg. C
    # -------------------------------
    avail_factor = max(0.0, 1.0 - 0.02 * dT)
    meta["gen_availability_factor"] = avail_factor

    # Dispatchable generators: cap max_p_mw 
    # (do not change current p_mw setpoint; OPF decides)
    if "gen" in net_scn and len(net_scn.gen):
        if "max_p_mw" not in net_scn.gen.columns:
            # fall back to current p_mw as a proxy cap if needed
            net_scn.gen["max_p_mw"] = net_scn.gen["p_mw"] if "p_mw" in net_scn.gen else 0.0
        net_scn.gen["max_p_mw"] = np.maximum(0.0, net_scn.gen["max_p_mw"] * avail_factor)

    # Static generators (e.g., PV/wind): scale available output
    if "sgen" in net_scn and len(net_scn.sgen):
        _scale(net_scn.sgen, "p_mw", avail_factor)
        if "q_mvar" in net_scn.sgen.columns:
            _scale(net_scn.sgen, "q_mvar", avail_factor)

    # --------------------------------------
    # 2) Higher operating costs (poly_cost)
    #    staff +50..100%, stock +10..20%
    # --------------------------------------
    staff_up = rng.uniform(0.50, 1.00)  # +50%..+100%
    stock_up = rng.uniform(0.10, 0.20)  # +10%..+20%
    cost_mult = 1.0 + staff_up + stock_up
    meta.update({"cost_multiplier": cost_mult, "staff_up": staff_up, "stock_up": stock_up})

    if "poly_cost" in net_scn and len(net_scn.poly_cost):
        pc = net_scn.poly_cost
        for c in pc.columns:
            cl = c.lower()
            if cl in ("cp1_eur_per_mw", "cp2_eur_per_mw2"):
                pc[c] = pc[c] * cost_mult

    # -------------------------------------------------
    # 3) System loss changes (thermal/ohmic effects)
    #    - Trafo capacity: -1%/C
    #    - Line resistance: +0.4%/C
    #    - Loss adder: +1% per 3 deg. C on demand
    # -------------------------------------------------
    # Transformers: derate max loading
    if "trafo" in net_scn and len(net_scn.trafo):
        if "max_loading_percent" not in net_scn.trafo.columns:
            net_scn.trafo["max_loading_percent"] = 100.0
        trafo_mult = max(0.0, 1.0 - 0.01 * dT)
        net_scn.trafo["max_loading_percent"] = np.maximum(10.0, net_scn.trafo["max_loading_percent"] * trafo_mult)
        meta["trafo_capacity_multiplier"] = trafo_mult

    # Lines: increase copper resistance
    if "line" in net_scn and len(net_scn.line):
        if "r_ohm_per_km" in net_scn.line.columns:
            r_mult = 1.0 + 0.004 * dT
            net_scn.line["r_ohm_per_km"] = net_scn.line["r_ohm_per_km"] * r_mult
            meta["line_resistance_multiplier"] = r_mult

    # Loss adder on demand: +1% per 3 deg C
    loss_adder = 1.0 + (dT / 3.0) * 0.01
    if "load" in net_scn and len(net_scn.load):
        _scale(net_scn.load, "p_mw", loss_adder)
        if "q_mvar" in net_scn.load.columns:
            _scale(net_scn.load, "q_mvar", loss_adder)
    meta["loss_adder_multiplier_on_load"] = loss_adder

    # --------------------------------
    # 4) High consumption (cap at 21%)
    #    linear with deltaT, capped at +0.21
    # --------------------------------
    # Assume ~3% per deg C up to +21% (i.e., 7 degC -> +21%)
    cons_up = min(0.21, 0.03 * dT)
    cons_mult = 1.0 + cons_up
    if "load" in net_scn and len(net_scn.load):
        _scale(net_scn.load, "p_mw", cons_mult)
        if "q_mvar" in net_scn.load.columns:
            _scale(net_scn.load, "q_mvar", cons_mult)
    meta["consumption_multiplier"] = cons_mult

    return net_scn, meta

def scenario_nk_contingencies(
    net,
    severity=1.0,
    seed=None,
    fraction_lines=0.1,
    fraction_gens=0.1,
    k_lines=None,
    k_gens=None,
    prefer_k= 1         #do we prefer using k values of fractions?
):
    """
    N-k contingencies: switch off multiple lines and/or generators.
    - severity scales the number of outages (used only if k_* is not set by scheme below).
    - control via fraction_* (share of total elements) or k_* (absolute).
    We prefer using small k (N-1 / N-2):
      (1) up to two generators removed
      (2) up to two lines removed
      (3) one line and one generator removed
      (4) no removal
    with all four events equally probable.
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)
    meta = {
        "category": "Topology/Contingency Stressors",
        "scenario": "N-k contingencies",
        "details": {},
    }

    # --------- prefer N-1 / N-2 k-values if both k_* are None ---------
    if prefer_k:
        n_lines = len(net_scn.line) if "line" in net_scn else 0
        n_gens = len(net_scn.gen) if "gen" in net_scn else 0

        event = rng.integers(0, 4)  # 0,1,2,3 with equal probability
        if event == 0:
            # (1) removal of up to two randomly chosen generators
            if n_gens > 0:
                k_gens = rng.integers(1, min(2, n_gens) + 1)
            else:
                k_gens = 0
            k_lines = 0
            meta["details"]["event"] = "gens_only"
        elif event == 1:
            # (2) removal of up to two randomly chosen lines
            if n_lines > 0:
                k_lines = rng.integers(1, min(2, n_lines) + 1)
            else:
                k_lines = 0
            k_gens = 0
            meta["details"]["event"] = "lines_only"
        elif event == 2:
            # (3) removal of a randomly chosen line and generator
            k_lines = 1 if n_lines > 0 else 0
            k_gens  = 1 if n_gens  > 0 else 0
            meta["details"]["event"] = "one_line_one_gen"
        else:
            # (4) no removal
            k_lines = 0
            k_gens  = 0
            meta["details"]["event"] = "none"

    # --------- line outages ----------
    if "line" in net_scn and len(net_scn.line):
        n_lines = len(net_scn.line)
        if k_lines is None:
            # fallback: base on fraction * severity
            k_lines_eff = max(0, int(np.round(fraction_lines * severity * n_lines)))
        else:
            k_lines_eff = int(k_lines)
        line_idx = _pick_indices(n_lines, k_lines_eff, rng)
        if "in_service" not in net_scn.line.columns:
            net_scn.line["in_service"] = True
        net_scn.line.iloc[line_idx, net_scn.line.columns.get_loc("in_service")] = False
        meta["details"]["line_outages"] = net_scn.line.index[line_idx].tolist()
    else:
        meta["details"]["line_outages"] = []

    # --------- generator outages ----------
    if "gen" in net_scn and len(net_scn.gen):
        n_gens = len(net_scn.gen)
        if k_gens is None:
            # fallback: base on fraction * severity
            k_gens_eff = max(0, int(np.round(fraction_gens * severity * n_gens)))
        else:
            k_gens_eff = int(k_gens)
        gen_idx = _pick_indices(n_gens, k_gens_eff, rng)
        if "in_service" not in net_scn.gen.columns:
            net_scn.gen["in_service"] = True
        net_scn.gen.iloc[gen_idx, net_scn.gen.columns.get_loc("in_service")] = False
        meta["details"]["gen_outages"] = net_scn.gen.index[gen_idx].tolist()
    else:
        meta["details"]["gen_outages"] = []

    return net_scn, meta


def scenario_reactive_scarcity(net, seed=None):
    """
    Reactive power scarcity:
    - Draw a random percentage x% in [0, 50].
    - Randomly switch off approximately x% of capacitor-like devices.
    - Both the percentage and the selected devices are random.
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)
    meta = {
        "category": "Reactive Power Stressors",
        "scenario": "Reactive power scarcity",
        "disabled_caps": [],
    }

    # Helper to apply random shutdown given a set of capacitor indices
    def _disable_random_caps(df, cap_idx, element_type):
        n_caps = len(cap_idx)
        if n_caps == 0:
            meta["note"] = f"no capacitor-like {element_type} elements found"
            return

        # 1) pick a random integer percentage between 0% and 50%
        percent = rng.integers(0, 51)  # [0, 50] inclusive
        frac = percent / 100.0

        # 2) compute how many to disable (rounded)
        k = int(np.round(frac * n_caps))
        k = min(max(k, 0), n_caps)  # clip to [0, n_caps]

        # If k == 0, we simply do not disable any elements (x% ~= 0)
        if k == 0:
            meta["element_type"] = element_type
            meta["target_percent"] = int(percent)
            meta["disabled_fraction"] = 0.0
            meta["disabled_count"] = 0
            return

        # 3) randomly choose which k elements to disable
        chosen = rng.choice(cap_idx, size=k, replace=False)

        if "in_service" not in df.columns:
            df["in_service"] = True
        df.loc[chosen, "in_service"] = False

        meta["element_type"] = element_type
        meta["target_percent"] = int(percent)
        meta["disabled_fraction"] = k / n_caps
        meta["disabled_count"] = int(k)
        meta["disabled_caps"] = chosen.tolist()

    # --------- Case 1: shunt elements (preferred) ----------
    if "shunt" in net_scn and len(net_scn.shunt):
        # Consider shunts with positive q_mvar as capacitors
        mask_cap = np.ones(len(net_scn.shunt), dtype=bool)
        if "q_mvar" in net_scn.shunt.columns:
            mask_cap &= net_scn.shunt["q_mvar"] > 0.0

        cap_idx = net_scn.shunt.index[mask_cap].to_numpy()
        if len(cap_idx):
            _disable_random_caps(net_scn.shunt, cap_idx, "shunt")
            return net_scn, meta

    # --------- Case 2: use sgens tagged as capacitors ----------
    if "sgen" in net_scn and len(net_scn.sgen):
        sgen = net_scn.sgen
        mask_cap = np.zeros(len(sgen), dtype=bool)

        if "type" in sgen.columns:
            type_str = sgen["type"].astype(str).str.lower()
            mask_cap |= type_str.str.contains("cap|capacitor|shunt")

        # fallback: any sgen with q_mvar > 0 but ~=zero p_mw
        if "q_mvar" in sgen.columns and "p_mw" in sgen.columns:
            mask_cap |= (sgen["q_mvar"] > 0.0) & (np.abs(sgen["p_mw"]) < 1e-4)

        cap_idx = sgen.index[mask_cap].to_numpy()
        if len(cap_idx):
            _disable_random_caps(net_scn.sgen, cap_idx, "sgen")
        else:
            meta["note"] = "no obvious capacitor banks found in shunt or sgen"
    else:
        meta["note"] = "no shunt or sgen elements found"

    return net_scn, meta


def scenario_islanded_operation(net, seed=None, max_cuts=3):
    """
    Islanded operation:
    - Open a random number of 'bridge-like' lines (between 0 and max_cuts, inclusive)
      to split the grid into islands.
    - Protocol:
        1) build a bus graph from in-service lines & trafos;
        2) form a BFS spanning tree;
        3) draw K ~ Uniform{0, ..., max_cuts};
        4) cut K edges from that tree where an actual line exists.
    """
    rng = _rng(seed)
    net_scn = copy.deepcopy(net)
    meta = {
        "category": "Topology Stressors",
        "scenario": "Islanded operation",
        "cut_lines": [],
    }

    if "bus" not in net_scn or len(net_scn.bus) == 0:
        meta["note"] = "no buses found"
        return net_scn, meta

    # --------- Build adjacency from in-service lines & trafos ----------
    n_bus = len(net_scn.bus)
    bus_indices = net_scn.bus.index.to_numpy()

    adj = {b: set() for b in bus_indices}

    # lines
    if "line" in net_scn and len(net_scn.line):
        line_df = net_scn.line
        if "in_service" not in line_df.columns:
            line_df["in_service"] = True
        active = line_df["in_service"].to_numpy()
        for idx, row in line_df[active].iterrows():
            fb, tb = row["from_bus"], row["to_bus"]
            if fb in adj and tb in adj:
                adj[fb].add(tb)
                adj[tb].add(fb)

    # trafos (also connect buses)
    if "trafo" in net_scn and len(net_scn.trafo):
        trafo_df = net_scn.trafo
        if "in_service" not in trafo_df.columns:
            trafo_df["in_service"] = True
        active = trafo_df["in_service"].to_numpy()
        for idx, row in trafo_df[active].iterrows():
            hb, lb = row["hv_bus"], row["lv_bus"]
            if hb in adj and lb in adj:
                adj[hb].add(lb)
                adj[lb].add(hb)

    # If graph is trivial, bail out
    if n_bus <= 1:
        meta["note"] = "single-bus system cannot be islanded meaningfully"
        return net_scn, meta

    # --------- BFS spanning tree ----------
    start_bus = rng.choice(bus_indices)
    visited = set([start_bus])
    parent = {start_bus: None}
    tree_edges = []  # (u, v) pairs

    queue = [start_bus]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                tree_edges.append((u, v))
                queue.append(v)

    if not tree_edges:
        meta["note"] = "graph is disconnected or trivial; no tree edges"
        return net_scn, meta

    # --------- choose a random number of tree edges to cut ----------
    # K from Uniform{0, ..., max_cuts}, but cannot exceed number of tree edges
    K = int(rng.integers(0, max_cuts + 1))  # inclusive upper bound
    K = min(K, len(tree_edges))
    meta["requested_cuts"] = int(K)

    if K == 0:
        meta["note"] = "random draw requested 0 cuts; topology unchanged"
        return net_scn, meta

    rng.shuffle(tree_edges)
    chosen_tree_edges = tree_edges[:K]

    cut_line_indices = []

    if "line" in net_scn and len(net_scn.line):
        line_df = net_scn.line
        if "in_service" not in line_df.columns:
            line_df["in_service"] = True

        for (u, v) in chosen_tree_edges:
            # find any line that connects u and v
            mask = (
                ((line_df["from_bus"] == u) & (line_df["to_bus"] == v)) |
                ((line_df["from_bus"] == v) & (line_df["to_bus"] == u))
            ) & line_df["in_service"]

            candidate_lines = line_df.index[mask].to_numpy()
            if len(candidate_lines):
                li = rng.choice(candidate_lines)
                line_df.loc[li, "in_service"] = False
                cut_line_indices.append(int(li))

    meta["cut_lines"] = cut_line_indices
    if not cut_line_indices:
        meta.setdefault("note", "no suitable lines found to cut; topology unchanged")

    return net_scn, meta

# ---------- scenario wrapper ----------
@register_scenario("load_extreme_high")
def _scn_load_extreme_high(net, seed=42, **kwargs):
    sev = random.uniform(0.5, 1.5)
    return scenario_load_level_extremes(net, severity=sev, seed=seed, mode="high")

@register_scenario("load_extreme_low")
def _scn_load_extreme_low(net, seed=42, **kwargs):
    sev = random.uniform(0.5, 1.5)
    return scenario_load_level_extremes(net, severity=sev, seed=seed, mode="low")

@register_scenario("renewable_variability")
def _scn_renewable_variability(net, seed=42, **kwargs):
    sev = random.uniform(0.5, 1.0)
    return scenario_renewable_variability(net, severity= sev, seed=seed)

@register_scenario("fuel_price_up")
def _scn_fuel_price_up(net, seed=42, **kwargs):
    sev = random.uniform(1.0, 2.0)
    return scenario_fuel_price_shocks(net, severity=sev, seed=seed, shock="up")

@register_scenario("fuel_price_down")
def _scn_fuel_price_down(net, seed=42, **kwargs):
    sev = random.uniform(1.0, 2.0)
    return scenario_fuel_price_shocks(net, severity=sev, seed=seed, shock="down")

@register_scenario("geo_shift")
def _scn_geo_shift(net, seed=42, **kwargs):
    return scenario_geographic_shift(net, severity=kwargs.get("severity", 1.0), seed=seed)

@register_scenario("extreme_weather")
def _scn_extreme_weather(net, seed=42, **kwargs):
    delc = random.randint(6, 20)
    return scenario_extreme_temperature(net, delta_c=delc, seed=seed)

@register_scenario("nk_contingencies")
def _scn_nk_contingencies(net, seed=42, **kwargs):
    return scenario_nk_contingencies(net, severity=kwargs.get("severity", 1.0), seed=seed)

@register_scenario("reactive_scarcity")
def _scn_reactive_scarcity(net, seed=42, **kwargs):
    return scenario_reactive_scarcity(net, seed=seed)

@register_scenario("islanded_operation")
def _scn_islanded_operation(net, seed=42, **kwargs):
    return scenario_islanded_operation(net, seed=seed, max_cuts=kwargs.get("max_cuts", 3))



def make_scenario(net, name, seed= 42, **kwargs):
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario name: {name}. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name](net, seed=seed, **kwargs)
