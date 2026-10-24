import numpy as np
import pandas as pd
from src.baselines.PINN.Get_KKT_Lg import Get_KKT_Lg

def create_data(net, simulation_parameters):
    n_bus= simulation_parameters['general']['n_bus']
    n_gbus = simulation_parameters['general']['n_gbus']
    n_line = simulation_parameters['general']['n_line']

    # -----------------------------
    # Load all inputs, then split into data + collocation
    # -----------------------------
    L_Val_all = pd.read_csv(f'src/baselines/PINN/datasets/{n_bus}/NN_input_train.csv').to_numpy()
    n_total = len(L_Val_all)
    if n_total < 2:
        raise ValueError("Not enough rows in NN_input_train.csv")

    # Use half labeled, half collocation  
    n_collocation = n_total // 2
    n_data_points = n_total - n_collocation

    # Inputs
    L_Val_data = L_Val_all[:n_data_points]       # (n_data_points, D)
    L_Val_collocation = L_Val_all[n_data_points:]       # (n_collocation, D)  (same distribution)
    L_Val = np.concatenate([L_Val_data, L_Val_collocation], axis=0)  # (n_total, D)

    # Type flag: 1 for data, 0 for collocation
    L_type_data = np.ones((n_data_points, 1), dtype=float)
    L_type_collocation = np.zeros((n_collocation, 1), dtype=float)
    L_type_total = np.concatenate([L_type_data, L_type_collocation], axis=0)  # (n_total, 1)

    x_training = [L_Val, L_type_total]  # both have n_total rows

    # -----------------------------
    # Labels: use only DATA rows from CSV, pad collocation with zeros
    # -----------------------------
    results_v_all= pd.read_csv(f'src/baselines/PINN/datasets/{n_bus}/NN_output_V_train.csv').to_numpy()
    results_pq_all = pd.read_csv(f'src/baselines/PINN/datasets/{n_bus}/NN_output_PQ_train.csv').to_numpy()

    if len(results_v_all) < n_data_points or len(results_pq_all) < n_data_points:
        raise ValueError(
            f"Output CSVs must have at least n_data_points={n_data_points} rows "
            f"(got V={len(results_v_all)}, PQ={len(results_pq_all)})"
        )

    results_v= results_v_all[:n_data_points]   # (n_data_points, 2*n_bus)
    results_pq = results_pq_all[:n_data_points]  # (n_data_points, 2*n_gbus)

    y_volt= np.concatenate(
        [results_v, np.zeros((n_collocation, 2*n_bus), dtype=float)], axis=0
    )
    y_PQ = np.concatenate(
        [results_pq, np.zeros((n_collocation, 2*n_gbus), dtype=float)], axis=0
    )

    # -----------------------------
    # KKT_Lg for DATA only, then pad collocation with zeros
    # -----------------------------
    KKT_Lg = Get_KKT_Lg(net, n_bus, L_Val_data, results_pq, results_v)  # (n_data_points, 4*n_bus+4*n_gbus+n_line)
    y_Lg = np.concatenate(
        [KKT_Lg, np.zeros((n_collocation, 4*n_bus + 4*n_gbus + n_line), dtype=float)], axis=0
    )

    # -----------------------------
    # Slice y_Lg into heads and normalize per-column
    # -----------------------------
    l_p_s, l_p_e = 0, 2*n_bus
    mu_g_u_s, mu_g_u_e = 2*n_bus, 2*n_bus + 2*n_gbus
    mu_g_d_s, mu_g_d_e = 2*n_bus + 2*n_gbus, 2*n_bus + 4*n_gbus
    mu_v_u_s, mu_v_u_e = 2*n_bus + 4*n_gbus, 3*n_bus + 4*n_gbus
    mu_v_d_s, mu_v_d_e = 3*n_bus + 4*n_gbus, 4*n_bus + 4*n_gbus
    mu_i_u_s, mu_i_u_e = 4*n_bus + 4*n_gbus, 4*n_bus + 4*n_gbus + n_line

    Lg_Max = []

    def _norm_and_store(Y, s, e):
        seg = Y[:, s:e]
        mx = np.maximum(np.max(np.abs(seg), axis=0, keepdims=True), 1e-5)  # (1, e-s)
        Lg_Max.append(mx)
        return seg / mx

    y_l_p_Lg = _norm_and_store(y_Lg, l_p_s,    l_p_e)
    y_mu_g_u_Lg = _norm_and_store(y_Lg, mu_g_u_s, mu_g_u_e)
    y_mu_g_d_Lg = _norm_and_store(y_Lg, mu_g_d_s, mu_g_d_e)
    y_mu_v_u_Lg = _norm_and_store(y_Lg, mu_v_u_s, mu_v_u_e)
    y_mu_v_d_Lg = _norm_and_store(y_Lg, mu_v_d_s, mu_v_d_e)
    y_mu_i_u_Lg = _norm_and_store(y_Lg, mu_i_u_s, mu_i_u_e)

    # Keep extra dummy head (ensure n_total rows)
    dummy_out = np.zeros((n_total, 1), dtype=float)

    y_training = [
        y_volt, y_PQ,
        y_l_p_Lg,
        y_mu_g_u_Lg, y_mu_g_d_Lg,
        y_mu_v_u_Lg, y_mu_v_d_Lg,
        y_mu_i_u_Lg,
        dummy_out,
    ]

    # -----------------------------
    # Sanity checks (all inputs/outputs same cardinality)
    # -----------------------------
    assert x_training[0].shape[0] == n_total and x_training[1].shape[0] == n_total, \
        f"x sizes: {x_training[0].shape[0]}, {x_training[1].shape[0]} vs n_total={n_total}"
    for i, y in enumerate(y_training):
        assert y.shape[0] == n_total, f"y[{i}] has {y.shape[0]} rows; expected {n_total}"

    # Save DATA portion only (as before)
    np.savetxt(f'src/baselines/PINN/test_output/{n_bus}/features_train.csv', L_Val_data, fmt='%s', delimiter=',')
    np.savetxt(f'src/baselines/PINN/test_output/{n_bus}/labels_train.csv',   results_pq, fmt='%s', delimiter=',')

    return x_training, y_training, Lg_Max
