# =========================
# DC evaluation parameters
# =========================

# Numerical cleanup threshold used inside DC helper calculations
dc_ctol = 1e-7

# Hard-feasibility tolerance for active-power balance in MW.
dc_balance_tol_mw = 1e-4

# Optional separate tolerances,in case we split feasibility checks
dc_gen_tol_mw = 1e-6
dc_flow_tol_mw = 1e-6

# Projection settings, True: project infeasible outputs, False: don't do projection
dc_project_if_needed = False

# Penalty cap used when a sample fails badly / crashes
dc_max_pen = 1e6