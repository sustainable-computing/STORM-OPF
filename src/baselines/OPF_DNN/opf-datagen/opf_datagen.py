#!/usr/bin/env python3
# pp_scenarios_to_json.py
import os, json, math, argparse
import numpy as np
import pandas as pd
from src import util

# ---------------- helpers ----------------
def _get_col(df, col, default=None):
    return df[col].to_numpy() if (df is not None and col in df.columns) else np.full(len(df) if df is not None else 0, default)

def _deg2rad(x):
    return float(x) * math.pi / 180.0

def _active_gen_indices(gen_df: pd.DataFrame) -> list[int]:
    """Active gens: pmax>0 (fallback to 1.2*p_now if pmax missing)."""
    if gen_df is None or not len(gen_df):
        return []
    pmax = gen_df["max_p_mw"].to_numpy() if "max_p_mw" in gen_df.columns else np.full(len(gen_df), np.nan)
    p_now = gen_df["p_mw"].to_numpy() if "p_mw" in gen_df.columns else np.zeros(len(gen_df))
    pmax = np.where(np.isfinite(pmax), pmax, np.maximum(1.0, p_now * 1.2))
    active = (pmax > 0.0)
    return [int(i) for i, a in zip(gen_df.index, active) if a]

def manifest_ok_paths(scen_dir: str):
    man = os.path.join(scen_dir, "manifest.csv")
    if not os.path.isfile(man):
        raise FileNotFoundError(f"Missing manifest.csv at {man}")
    df = pd.read_csv(man)
    if "status" not in df or "file" not in df:
        raise ValueError("manifest.csv must contain 'status' and 'file' columns.")
    df = df[df["status"] == "ok"].reset_index(drop=True)
    return [os.path.join(scen_dir, f) for f in df["file"].tolist()]

# ---------------- constraints from a reference net ----------------
def constraints_from_reference_net(net):
    """
    Build constraints dict:
      pg_lim/qg_lim keyed by generator index (string) for gens with pmax>0,
      vm_lim keyed by bus index (string),
      rate_a keyed by branches ("L<idx>" lines, "T<idx>" trafos),
      line_rx/bg placeholders as before.
    """
    vg_lim, pg_lim, qg_lim = {}, {}, {}
    vm_lim = {}

    # bus voltage limits
    bus = net.bus if len(getattr(net, "bus", [])) else pd.DataFrame(index=[])
    vmin = _get_col(bus, "min_vm_pu", 0.95)
    vmax = _get_col(bus, "max_vm_pu", 1.05)
    for i, (lo, hi) in zip(bus.index, zip(vmin, vmax)):
        vm_lim[str(i)] = (float(lo), float(hi))

    # gens
    gen = net.gen if len(getattr(net, "gen", [])) else pd.DataFrame(index=[])
    if len(gen):
        pmin = _get_col(gen, "min_p_mw", 0.0)
        pmax = _get_col(gen, "max_p_mw", np.nan)
        qmin = _get_col(gen, "min_q_mvar", -1e9)
        qmax = _get_col(gen, "max_q_mvar",  1e9)
        p_now = _get_col(gen, "p_mw", 0.0)
        pmax = np.where(np.isfinite(pmax), pmax, np.maximum(1.0, p_now * 1.2))
        for gi, lo, hi, qlo, qhi, bus_i in zip(gen.index, pmin, pmax, qmin, qmax, gen["bus"].to_numpy()):
            if hi > 0.0:
                pg_lim[str(gi)] = (float(lo), float(hi))
                qg_lim[str(gi)] = (float(qlo), float(qhi))
                v_lo, v_hi = vm_lim[str(int(bus_i))]
                vg_lim[str(gi)] = (float(v_lo), float(v_hi))

    # branch ratings & params
    rate_a, line_rx, line_bg = {}, {}, {}

    # lines
    if len(getattr(net, "line", [])):
        line = net.line
        vn_kv = _get_col(net.bus.loc[line.from_bus.values, :], "vn_kv", 0.0)
        imax  = _get_col(line, "max_i_ka", np.nan)
        rate_mva = np.sqrt(3.0) * vn_kv * np.nan_to_num(imax, nan=0.0)
        rate_mva = np.where(rate_mva <= 0.0, 1e6, rate_mva)  # big fallback
        r_ohm_km = _get_col(line, "r_ohm_per_km", 0.0)
        x_ohm_km = _get_col(line, "x_ohm_per_km", 0.0)
        length   = _get_col(line, "length_km", 0.0)
        r_series = r_ohm_km * length
        x_series = x_ohm_km * length
        for li, rate, r, x in zip(line.index, rate_mva, r_series, x_series):
            key = f"L{li}"
            rate_a[key] = float(rate)
            line_rx[key] = (float(r), float(x))
            line_bg[key] = (0.0, 0.0)

    # trafos
    if len(getattr(net, "trafo", [])):
        trafo = net.trafo
        sn = _get_col(trafo, "sn_mva", np.nan)
        sn = np.where(np.isfinite(sn), sn, 1e6)
        for ti, s in zip(trafo.index, sn):
            key = f"T{ti}"
            rate_a[key] = float(s)
            line_rx[key] = (0.0, 0.0)
            line_bg[key] = (0.0, 0.0)

    return {"vg_lim": vg_lim, "pg_lim": pg_lim, "qg_lim": qg_lim,
            "vm_lim": vm_lim, "rate_a": rate_a, "line_rx": line_rx, "line_bg": line_bg}

# ---------------- per-scenario packing ----------------
def experiment_from_net(net, base_pd_by_load=None):
    """
    Per-experiment dict with keys identical to the Julia baseline:
      pd/qd are keyed by load index (strings),
      pg/qg/vg keyed by generator index (active gens only),
      pf/pt/qf/qt keyed by "L<i>" lines then "T<i>" trafos,
      vm/va keyed by bus index (strings; va in radians).
    """
    # inputs (loads)
    pd, qd = {}, {}
    if len(getattr(net, "load", [])):
        for i, row in net.load.iterrows():
            pd[str(i)] = float(row.get("p_mw", 0.0))
            qd[str(i)] = float(row.get("q_mvar", 0.0))

    # scale relative to reference loads (optional)
    if base_pd_by_load and len(base_pd_by_load) > 0:
        ratios = []
        for k, base_pd in base_pd_by_load.items():
            cur = pd.get(k, 0.0)
            if base_pd > 0.0:
                ratios.append(cur / base_pd)
        scale = float(np.mean(ratios)) if ratios else 1.0
    else:
        scale = 1.0

    # generator outputs (active gens)
    pg, qg, vg = {}, {}, {}
    if len(getattr(net, "gen", [])):
        pmax = _get_col(net.gen, "max_p_mw", np.nan)
        p_now = _get_col(net.gen, "p_mw", 0.0)
        pmax = np.where(np.isfinite(pmax), pmax, np.maximum(1.0, p_now * 1.2))
        active_mask = pmax > 0.0
        for (gi, grow), is_active in zip(net.gen.iterrows(), active_mask):
            if not is_active:
                continue
            if len(getattr(net, "res_gen", [])):
                pg[str(gi)] = float(net.res_gen.at[gi, "p_mw"]) if "p_mw" in net.res_gen.columns else 0.0
                qg[str(gi)] = float(net.res_gen.at[gi, "q_mvar"]) if "q_mvar" in net.res_gen.columns else 0.0
            else:
                pg[str(gi)] = 0.0
                qg[str(gi)] = 0.0
            bus_i = int(grow["bus"])
            vm_val = float(net.res_bus.at[bus_i, "vm_pu"]) if len(getattr(net, "res_bus", [])) else 1.0
            vg[str(gi)] = vm_val

    # flows
    pf, pt, qf, qt = {}, {}, {}, {}
    if len(getattr(net, "res_line", [])):
        for i, row in net.res_line.iterrows():
            key = f"L{i}"
            pf[key] = float(row.get("p_from_mw", 0.0))
            pt[key] = float(row.get("p_to_mw",   0.0))
            qf[key] = float(row.get("q_from_mvar", 0.0))
            qt[key] = float(row.get("q_to_mvar",   0.0))
    if len(getattr(net, "res_trafo", [])):
        for i, row in net.res_trafo.iterrows():
            key = f"T{i}"
            pf[key] = float(row.get("p_hv_mw", 0.0))
            pt[key] = float(row.get("p_lv_mw", 0.0))
            qf[key] = float(row.get("q_hv_mvar", 0.0))
            qt[key] = float(row.get("q_lv_mvar", 0.0))

    # buses
    va, vm = {}, {}
    if len(getattr(net, "res_bus", [])):
        for i, row in net.res_bus.iterrows():
            vm[str(i)] = float(row.get("vm_pu", 1.0))
            va[str(i)] = _deg2rad(float(row.get("va_degree", 0.0)))

    return {
        "scale": float(scale),
        "pd": pd, "qd": qd,
        "vg": vg, "pg": pg, "qg": qg,
        "pt": pt, "pf": pf, "qt": qt, "qf": qf,
        "va": va, "vm": vm,
        "objective": 0.0, "solve_time": 0.0,
    }

def build_meta_and_order(ref_net, case, scen_paths):
    """Produce meta + explicit canonical order lists used by Julia."""
    bus_order  = [int(i) for i in ref_net.bus.index.tolist()]
    load_order = [int(i) for i in ref_net.load.index.tolist()] if len(getattr(ref_net, "load", [])) else []
    gen_order  = _active_gen_indices(ref_net.gen if len(getattr(ref_net, "gen", [])) else None)

    # flow order: L<i> then T<i>
    line_order  = [f"L{i}" for i in (ref_net.line.index.tolist() if len(getattr(ref_net, "line", [])) else [])]
    trafo_order = [f"T{i}" for i in (ref_net.trafo.index.tolist() if len(getattr(ref_net, "trafo", [])) else [])]
    flow_order  = line_order + trafo_order

    n_vm = len(bus_order)
    n_va = n_vm
    n_pg = len(gen_order)
    slices = {
        "vm": (0, n_vm),
        "va": (n_vm, n_vm + n_va),
        "pg": (n_vm + n_va, n_vm + n_va + n_pg),
    }
    meta = {
        "case": case,
        "units": {"vm": "pu", "va": "rad", "p": "MW", "q": "Mvar"},
        "Sbase_MVA": float(getattr(ref_net, "sn_mva", 100.0)),
        "bus_index_order": bus_order,     # for bus-wise arrays (vm/va)
        "gen_index_order": gen_order,     # for per-gen arrays (pg/qg/vg)
        "dims": {"n_vm": n_vm, "n_va": n_va, "n_pg": n_pg, "y_dim": n_vm + n_va + n_pg},
        "slices": slices,
        "net_paths": list(scen_paths),
    }
    # Explicit orders for *all* dict-shaped tensors
    order = {
        "pd":  [str(i) for i in load_order],
        "qd":  [str(i) for i in load_order],
        "vm":  [str(i) for i in bus_order],
        "va":  [str(i) for i in bus_order],
        "pg":  [str(i) for i in gen_order],
        "qg":  [str(i) for i in gen_order],
        "pf":  flow_order,
        "qf":  flow_order,
        "pt":  flow_order,
        "qt":  flow_order,
    }
    return meta, order

# ---------------- JSON builders ----------------
def build_json_from_paths(scen_paths, case, which_data, outfile):
    if not scen_paths:
        raise RuntimeError("No valid scenario files found.")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # reference net (constraints + order)
    ref_net, _ = util.get_base_net(case, 'in_distribution')

    # base loads for scale
    base_pd_by_load = {}
    if len(getattr(ref_net, "load", [])):
        for i, row in ref_net.load.iterrows():
            base_pd_by_load[str(i)] = float(row.get("p_mw", 0.0))

    constraints = constraints_from_reference_net(ref_net)
    meta, order = build_meta_and_order(ref_net, case, scen_paths)

    experiments = []
    for p in scen_paths:
        #print('new net')
        net = util.load_net(p)
        exp = experiment_from_net(net, base_pd_by_load=base_pd_by_load)
        experiments.append(exp)

    out = {
        "experiments": experiments,
        "constraints": constraints,
        "meta": meta,
        "order": order,            # <<< IMPORTANT: explicit canonical order for Julia
        "net_paths": scen_paths,   # dup for convenience
    }
    with open(outfile, "w") as f:
        json.dump(out, f, indent=4)
    print(f"Writing: {outfile}")

# ---------------- split helpers ----------------
def build_training_json_from_scenarios(*, case: str, scen_root: str, op_file_path: str):
    scen_dir = os.path.join(scen_root, case, "training")
    scen_paths = manifest_ok_paths(scen_dir)
    build_json_from_paths(scen_paths, case, 'in_distribution', op_file_path)

def nested_manifest_ok_paths(scen_root_dir: str):
    if not os.path.isdir(scen_root_dir):
        raise FileNotFoundError(f"Missing directory: {scen_root_dir}")

    group_dirs = [
        os.path.join(scen_root_dir, d)
        for d in sorted(os.listdir(scen_root_dir))
        if os.path.isdir(os.path.join(scen_root_dir, d))
    ]
    if len(group_dirs) == 0:
        raise RuntimeError(f"No stress-case folders found under: {scen_root_dir}")

    all_paths = []
    for group_dir in group_dirs:
        man = os.path.join(group_dir, "manifest.csv")
        if not os.path.isfile(man):
            raise FileNotFoundError(f"Missing manifest.csv at {man}")

        df = pd.read_csv(man)
        if "status" not in df or "file" not in df:
            raise ValueError(f"manifest.csv must contain 'status' and 'file' columns: {man}")

        df = df[df["status"] == "ok"].reset_index(drop=True)

        for f in df["file"].tolist():
            p = os.path.join(group_dir, str(f))
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Scenario file listed in manifest not found: {p}")
            all_paths.append(p)

    if len(all_paths) == 0:
        raise RuntimeError(f"No valid scenario files found under: {scen_root_dir}")

    np.random.shuffle(all_paths)
    return all_paths

def build_stress_training_json_from_scenarios(*, case: str, scen_root: str, op_file_path: str):
    scen_dir = os.path.join(scen_root, case, "training_stressed")
    scen_paths = nested_manifest_ok_paths(scen_dir)
    build_json_from_paths(scen_paths, case, "training_stressed", op_file_path)


def build_test_json_from_scenarios(*, case: str, which_data: str, scen_root: str,
                                   output_filepath: str):
    assert which_data, "which_data must be provided for test split"
    scen_dir = os.path.join(scen_root, case, "test", which_data)
    scen_paths = manifest_ok_paths(scen_dir)
    build_json_from_paths(scen_paths, case, which_data, output_filepath)

# ---------------- CLI ----------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Export pandapower scenario nets to Julia-compatible JSON.")
    ap.add_argument("--case", "-c", type=str, default='case14')
    ap.add_argument("--split", choices=["training", "training_stressed", "test"], default="training")
    ap.add_argument("--which_data", type=str, default="in_distribution")
    ap.add_argument("--scen_root", type=str, default="src/scenarios")
    ap.add_argument("--netname_label", type=str, default=None)
    return ap.parse_args()

def main():
    args = _parse_args()
    
    if args.split == "training":
        output = f'src/baselines/OPF_DNN/datasets/{args.case}/{args.split}_data_{args.which_data}.json'
        build_training_json_from_scenarios(case=args.case, scen_root=args.scen_root, op_file_path=output)
    
    elif args.split == "training_stressed":
        output = f"src/baselines/OPF_DNN/datasets/{args.case}/training_data_stressed.json"
        build_stress_training_json_from_scenarios(case=args.case, scen_root=args.scen_root, op_file_path=output)

    else:
        if not args.which_data:
            raise SystemExit("For --split test you must provide --which_data")
        output = f'src/baselines/OPF_DNN/datasets/{args.case}/{args.split}_data_{args.which_data}.json'
        build_test_json_from_scenarios(case=args.case, which_data=args.which_data, scen_root=args.scen_root, output_filepath=output)

if __name__ == "__main__":
    main()
