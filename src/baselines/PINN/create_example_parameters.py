import numpy as np
import pandapower as pp
import pandas as pd

def create_example_parameters(net):
    """
    Adapted from the original PINN setup, but we're using pandapower to extract system parameters.
    """
    # -----------------------------
    # Ensure Ybus and internal ppc are built
    # -----------------------------
    try:
        pp.runpp(net, calculate_voltage_angles=True, init="flat", numba=True)
    except Exception:
        pp.runpp(net, calculate_voltage_angles=False, init="auto", numba=False, algorithm="nr")
    if not hasattr(net, "_ppc"):
        raise RuntimeError("pandapower could not build an internal ppc for this net.")

    ppc = net._ppc
    internal = ppc.get("internal", {})
    if not internal:
        raise RuntimeError("pandapower internal matrices (Ybus, Yf, Yt) not available in net._ppc['internal'].")

    # -----------------------------
    # Bus / branch cardinalities & mappings (0-based indexing)
    # -----------------------------
    n_bus = len(net.bus)

    # Branch list: lines then trafos, keep order
    line_from = net.line.from_bus.to_numpy() if len(net.line) else np.array([], dtype=int)
    line_to = net.line.to_bus.to_numpy()   if len(net.line) else np.array([], dtype=int)

    trafo_from = net.trafo.hv_bus.to_numpy() if len(net.trafo) else np.array([], dtype=int)
    trafo_to = net.trafo.lv_bus.to_numpy() if len(net.trafo) else np.array([], dtype=int)

    fbus = np.concatenate([line_from, trafo_from]).astype(int)
    tbus = np.concatenate([line_to,   trafo_to]).astype(int)
    n_line = fbus.size  # total branches (lines + trafos)

    # -----------------------------
    # Incidence matrix IM  (2*n_line × 2*n_bus)
    # -----------------------------
    if n_line > 0:
        B = np.zeros((n_line, n_bus), dtype=float)
        rows = np.arange(n_line)
        B[rows, fbus] -= 1.0
        B[rows, tbus] += 1.0
    else:
        B = np.zeros((0, n_bus), dtype=float)

    Z = np.zeros_like(B)
    IM = np.vstack([
        np.hstack([B, Z]),
        np.hstack([Z, B])
    ])
    assert IM.shape == (2*n_line, 2*n_bus), f"IM shape {IM.shape} != {(2*n_line, 2*n_bus)}"

    # -----------------------------
    # System base MVA (used in several places; define early)
    # -----------------------------
    Sbase = float(getattr(net, "sn_mva", 100.0))

    # -----------------------------
    # Ybus -> real-augmented 2N×2N
    # -----------------------------
    Ybus_csr = internal.get("Ybus", None)
    if Ybus_csr is None:
        raise RuntimeError("Ybus not found in net._ppc['internal'].")

    Yc = Ybus_csr.toarray()  # complex (N × N)
    Yr = np.real(Yc)
    Yi = np.imag(Yc)

    def real_augment(A_re, A_im):
        # [A_re -A_im; A_im A_re]
        return np.block([[A_re, -A_im],
                         [A_im,  A_re]])

    Y     = real_augment(Yr,  Yi)     # (2N × 2N)
    Yconj = real_augment(Yr, -Yi)     # conj(Yc) in real-augmented form
    assert Y.shape == (2*n_bus, 2*n_bus), f"Y shape {Y.shape} != {(2*n_bus, 2*n_bus)}"

    # -----------------------------
    # Branch series admittance matrix Ybr (2*n_line × 2*n_line, block-diagonal)
    # One 2×2 block [[g,-b],[b,g]] per branch (line or trafo) in our order (lines then trafos).
    # -----------------------------
    def _safe_inv(z):
        eps = 1e-12
        return 1.0 / (z + eps)

    y_lines = np.array([], dtype=complex)
    if len(net.line):
        # Z_base(ohm) = (V_kV^2) / Sbase(MVA)
        Vbase_kV = net.bus.loc[net.line.from_bus, "vn_kv"].to_numpy()
        Zbase_ohm = (Vbase_kV ** 2) / Sbase
        r_ohm = (net.line["r_ohm_per_km"].to_numpy() * net.line["length_km"].to_numpy())
        x_ohm = (net.line["x_ohm_per_km"].to_numpy() * net.line["length_km"].to_numpy())
        z_pu = (r_ohm + 1j * x_ohm) / Zbase_ohm
        y_lines= _safe_inv(z_pu)

    y_trafos = np.array([], dtype=complex)
    if len(net.trafo):
        # Use vk%, vkr% on trafo base, then convert to system base
        vk = net.trafo.get("vk_percent",  pd.Series(np.full(len(net.trafo), 10.0))).to_numpy() / 100.0
        vkr = net.trafo.get("vkr_percent", pd.Series(np.full(len(net.trafo),  0.5))).to_numpy() / 100.0
        sn = net.trafo.get("sn_mva",      pd.Series(np.full(len(net.trafo),  Sbase))).to_numpy()
        r_pu_tr = vkr
        x_pu_tr = np.sqrt(np.maximum(vk**2 - r_pu_tr**2, 0.0))
        base_factor = Sbase / sn
        r_pu = r_pu_tr * base_factor
        x_pu = x_pu_tr * base_factor
        z_pu_tr = r_pu + 1j * x_pu
        y_trafos = _safe_inv(z_pu_tr)

    if n_line > 0:
        y_all = np.concatenate([y_lines, y_trafos]).astype(complex)
    else:
        y_all = np.array([], dtype=complex)

    Ybr = np.zeros((2*n_line, 2*n_line), dtype=float)
    for k in range(n_line):
        g = float(np.real(y_all[k])) if y_all.size else 0.0
        b = float(np.imag(y_all[k])) if y_all.size else 0.0
        Ybr[2*k:2*k+2, 2*k:2*k+2] = np.array([[ g, -b],
                                              [ b,  g]], dtype=float)
    assert Ybr.shape == (2*n_line, 2*n_line), f"Ybr shape {Ybr.shape} != {(2*n_line, 2*n_line)}"

    # -----------------------------
    # Generators (include slack via ext_grid) and loads
    # -----------------------------
    gen_df = net.gen.copy()
    ext_df = net.ext_grid.copy()
    # reindex ext_grid to avoid index collisions before concat
    if len(gen_df):
        start = int(gen_df.index.max()) + 1
    else:
        start = 0
    ext_df.index = range(start, start + len(ext_df))
    gen_df = pd.concat([gen_df, ext_df], axis=0)

    g_bus = gen_df.bus.to_numpy().astype(int) if len(gen_df) else np.array([], dtype=int)
    n_gbus = g_bus.size

    load_df = net.load.copy()
    l_bus = np.unique(load_df.bus.to_numpy().astype(int)) if len(load_df) else np.array([], dtype=int)
    n_lbus = l_bus.size

    # -----------------------------
    # Voltage limits
    # -----------------------------
    Vmin = float(np.nanmin(net.bus["min_vm_pu"].fillna(0.94))) if "min_vm_pu" in net.bus.columns and len(net.bus) else 0.94
    Vmax = float(np.nanmax(net.bus["max_vm_pu"].fillna(1.06))) if "max_vm_pu" in net.bus.columns and len(net.bus) else 1.06

    # -----------------------------
    # Generator P/Q limits (p.u. on Sbase)
    # -----------------------------
    def get_or_default(df, col, default_val):
        return df[col].to_numpy() if (len(df) and col in df.columns) else np.full(len(df), default_val, dtype=float)

    Pg_min_MW = get_or_default(gen_df, "min_p_mw", 0.0)
    Pg_max_MW = gen_df["max_p_mw"].to_numpy() if (len(gen_df) and "max_p_mw" in gen_df.columns) else (
        gen_df["p_mw"].to_numpy() * 1.2 if (len(gen_df) and "p_mw" in gen_df.columns) else np.full(n_gbus, 0.0)
    )
    Qg_min_MVAr = get_or_default(gen_df, "min_q_mvar", -1e3)
    Qg_max_MVAr = get_or_default(gen_df, "max_q_mvar",  1e3)

    Pg_min = (Pg_min_MW  / Sbase).reshape(1, n_gbus) if n_gbus else np.zeros((1, 0))
    Pg_max = (Pg_max_MW  / Sbase).reshape(1, n_gbus) if n_gbus else np.zeros((1, 0))
    Qg_min = (Qg_min_MVAr / Sbase).reshape(1, n_gbus) if n_gbus else np.zeros((1, 0))
    Qg_max = (Qg_max_MVAr / Sbase).reshape(1, n_gbus) if n_gbus else np.zeros((1, 0))

    Gen_max = np.concatenate([Pg_max, Qg_max], axis=1).reshape(1, 2*n_gbus) if n_gbus else np.zeros((1, 0))
    Gen_min = np.concatenate([Pg_min, Qg_min], axis=1).reshape(1, 2*n_gbus) if n_gbus else np.zeros((1, 0))

    # -----------------------------
    # Linear gen cost (row vector)
    # -----------------------------
    if hasattr(net, "poly_cost") and len(net.poly_cost):
        pc = net.poly_cost
        mask = (pc.et == "gen") if "et" in pc.columns else np.zeros(len(pc), dtype=bool)
        C_map = {}
        if mask.any():
            pc_gen = pc[mask]
            if "cp1_eur_per_mw" in pc_gen.columns:
                for _, row in pc_gen.iterrows():
                    C_map[int(row["element"])] = float(row["cp1_eur_per_mw"])
            elif "cp1" in pc_gen.columns:
                for _, row in pc_gen.iterrows():
                    C_map[int(row["element"])] = float(row["cp1"])
        C_list = [C_map.get(int(idx), 1.0) for idx in gen_df.index] if len(gen_df) else []
        C_Pg = np.asarray(C_list, dtype=float).reshape(1, n_gbus) if n_gbus else np.zeros((1, 0))
    else:
        C_Pg = np.ones((1, n_gbus), dtype=float) if n_gbus else np.zeros((1, 0))

    # -----------------------------
    # Map_g and Map_L (real-augmented)
    # -----------------------------
    Map_g = np.zeros((2*n_gbus, 2*n_bus), dtype=float)
    for gi, b in enumerate(g_bus):
        Map_g[gi, b] = 1.0
        Map_g[n_gbus + gi, n_bus + b] = 1.0

    Map_L = np.zeros((2*n_lbus, 2*n_bus), dtype=float)
    for li, b in enumerate(l_bus):
        Map_L[li, b] = 1.0
        Map_L[n_lbus + li, n_bus + b] = 1.0

    # -----------------------------
    # Branch limits (S_max p.u.) 1 × n_line
    # -----------------------------
    L_limit = np.zeros((1, n_line), dtype=float)
    if len(net.line):
        Imax_line = net.line["max_i_ka"].to_numpy() if "max_i_ka" in net.line.columns else np.full(len(net.line), 1e3, dtype=float)
        Vbase_line = net.bus.loc[net.line.from_bus, "vn_kv"].to_numpy()
        Smax_line_MVA = np.sqrt(3.0) * Vbase_line * Imax_line
        L_limit[0, :len(net.line)] = Smax_line_MVA / Sbase

    if len(net.trafo):
        if "sn_mva" in net.trafo.columns:
            Smax_trafo_MVA = net.trafo["sn_mva"].to_numpy()
        else:
            Smax_trafo_MVA = np.full(len(net.trafo), 1e3, dtype=float)
        L_limit[0, len(net.line):] = Smax_trafo_MVA / Sbase

    # -----------------------------
    # Sanity asserts for the key trio
    # -----------------------------
    assert Y.shape == (2*n_bus, 2*n_bus)
    assert IM.shape == (2*n_line, 2*n_bus)
    assert Ybr.shape == (2*n_line, 2*n_line)

    # -----------------------------
    # Package outputs
    # -----------------------------
    true_system_parameters = {
        'Gen_max': Gen_max,
        'Gen_min': Gen_min,
        'Vmin': Vmin,
        'Vmax': Vmax,
        'L_limit': L_limit,
        'Y': Y,
        'Yconj': Yconj,
        'Ybr': Ybr,
        'IM': IM,
        'C_Pg': C_Pg,
        'Map_g': Map_g,
        'Map_L': Map_L
    }

    general_parameters = {
        'n_bus': n_bus,
        'n_lbus': n_lbus,
        'n_gbus': n_gbus,
        'n_line': n_line,
        'l_bus': l_bus.astype(int),
        'g_bus': g_bus.astype(int)
    }

    # Training/data knobs 
    n_data_points = 5000
    n_collocation_points = 5000
    epoch_schedule = [1000]
    n_total = n_data_points + n_collocation_points
    batching_schedule = [int(np.ceil(n_total / 200))]

    data_creation_parameters = {
        'n_data_points': n_data_points,
        'n_collocation': n_collocation_points
    }
    training_parameters = {
        'epoch_schedule': epoch_schedule,
        'batching_schedule': batching_schedule,
        'neurons_in_hidden_layers_V': [40, 40, 40],
        'neurons_in_hidden_layers_G': [20, 20, 20],
        'neurons_in_hidden_layers_Lg': [50, 50, 50]
    }

    simulation_parameters = {
        'true_system': true_system_parameters,
        'general': general_parameters,
        'data_creation': data_creation_parameters,
        'training': training_parameters
    }
    return simulation_parameters
