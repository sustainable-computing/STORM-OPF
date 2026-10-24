"""
Curriculum training for the extended-input PINN model.

Minimal delta from train_ext_input.py:
  + targeted stressed dataset (TARGET_GROUPS only) built via train_augmented machinery
  + curriculum loop (warmup / ramp) over nominal + stressed
  + warm-start from a trained extended model
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src import util
from src.baselines.PINN import generate_data
from src.baselines.PINN.create_data import create_data
from src.baselines.PINN.create_example_parameters import create_example_parameters

# Extended model classes and norm-stats workers from train_ext_input
from src.baselines.PINN.train_ext_input import (
    PinnLayerExt,
    PinnModelExt,
    _build_norm_stats,
    _load_paths,
    _mp_context,
    _norm_worker,
    _norm_worker_init,
)

# Curriculum utilities from train_augmented
from src.baselines.PINN.train_augmented import (
    _apply_existing_stats,
    _assert_stress_cache_exists,
    _dataset_dir,
    _nrows,
    _temporary_train_csv_swap,
    _to_numpy_list,
    _train_paths,
    get_epoch_stress_count,
    sample_epoch_dataset_full_union,
)

TARGET_GROUPS = ["geo_shift", "extreme_weather", "load_extreme_high", "load_extreme_low"]


# -----------------------------------------------------------------------
# Targeted stress cache: only TARGET_GROUPS (not all stressed groups)
# -----------------------------------------------------------------------

def _target_cache_paths(num_bus: str):
    d = _dataset_dir(num_bus)
    return {
        "in":   os.path.join(d, "NN_input_train_stressed_targeted.csv"),
        "pq":   os.path.join(d, "NN_output_PQ_train_stressed_targeted.csv"),
        "v":    os.path.join(d, "NN_output_V_train_stressed_targeted.csv"),
        "meta": os.path.join(d, "training_stressed_targeted_meta.json"),
    }


def _build_target_stress_cache(case: str):
    """Like train_augmented._build_stress_raw_dataframes but restricted to TARGET_GROUPS."""
    num_bus = case[4:]
    base = _train_paths(num_bus)
    for p in (base["stats_in"], base["stats_pq"], base["stats_v"]):
        if not os.path.exists(p):
            raise FileNotFoundError("Required baseline norm stat not found: {}".format(p))

    scen_root = os.path.join("src", "scenarios", case, "training_stressed")
    rows_in, rows_pq, rows_v, kept_groups = [], [], [], []
    skipped = 0
    group_counts = {}

    for group in TARGET_GROUPS:
        group_dir = os.path.join(scen_root, group)
        if not os.path.isdir(group_dir):
            print("[target-cache] WARNING: group dir not found, skipping: {}".format(group_dir))
            continue
        mf_path = os.path.join(group_dir, "manifest.csv")
        if not os.path.exists(mf_path):
            print("[target-cache] WARNING: missing manifest, skipping: {}".format(mf_path))
            continue
        mf = pd.read_csv(mf_path)
        mf = mf[mf["status"] == "ok"].reset_index(drop=True)
        group_counts[group] = int(len(mf))

        for rel_file in mf["file"].tolist():
            path = os.path.join(group_dir, str(rel_file))
            if not os.path.exists(path):
                skipped += 1
                continue
            ncopy = util.load_net(path)
            try:
                generate_data.run_opf(ncopy)
                if not getattr(ncopy, "OPF_converged", False):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            rows_in.append(generate_data._collect_inputs_from_loads(ncopy))
            rows_pq.append(generate_data._collect_outputs_gen_pq(ncopy))
            rows_v.append(generate_data._collect_outputs_v(ncopy))
            kept_groups.append(group)

    if not rows_in:
        raise RuntimeError("No feasible targeted stressed samples found.")

    df_in = pd.DataFrame(rows_in)
    df_pq = pd.DataFrame(rows_pq)
    df_v  = pd.DataFrame(rows_v)
    df_in = df_in.reindex(generate_data._natural_sort(df_in.columns), axis=1)
    df_pq = df_pq.reindex(generate_data._natural_sort(df_pq.columns), axis=1)
    df_v  = df_v.reindex(generate_data._natural_sort(df_v.columns), axis=1)

    perm = np.random.permutation(len(df_in))
    df_in = df_in.iloc[perm].reset_index(drop=True)
    df_pq = df_pq.iloc[perm].reset_index(drop=True)
    df_v  = df_v.iloc[perm].reset_index(drop=True)

    cache = _target_cache_paths(num_bus)
    os.makedirs(_dataset_dir(num_bus), exist_ok=True)

    _apply_existing_stats(df_in, base["stats_in"]).to_csv(cache["in"], index=False)
    _apply_existing_stats(df_pq, base["stats_pq"]).to_csv(cache["pq"], index=False)
    _apply_existing_stats(df_v,  base["stats_v"]).to_csv(cache["v"],  index=False)

    kept_per_group = {}
    for g in kept_groups:
        kept_per_group[g] = kept_per_group.get(g, 0) + 1
    print("[target-cache] groups: {}  total_kept={}  skipped={}".format(
        kept_per_group, sum(kept_per_group.values()), skipped))


def _assert_target_cache_exists(case: str):
    cache = _target_cache_paths(case[4:])
    for p in (cache["in"], cache["pq"], cache["v"]):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "Missing targeted stress cache: {}  Run with --gen_data first.".format(p))


def _load_target_stress_arrays(net, simulation_parameters, case: str):
    """Load targeted-group stressed training arrays via the CSV-swap trick (same as train_augmented)."""
    num_bus = case[4:]
    cache = _target_cache_paths(num_bus)
    with _temporary_train_csv_swap(num_bus, cache):
        x_training, y_training, Lg_Max = create_data(
            net, simulation_parameters=copy.deepcopy(simulation_parameters))
    return _to_numpy_list(x_training), _to_numpy_list(y_training), [np.asarray(x) for x in Lg_Max]


# -----------------------------------------------------------------------
# Extended-feature helpers
# -----------------------------------------------------------------------

def _make_extended_inputs(L_val_orig, P_mu, P_sig, nominal_total, n_lbus, n_gen, n_trafo):
    """Add p_frac / log_scale / ratio placeholders to a normalized load matrix."""
    n = L_val_orig.shape[0]
    P_D = L_val_orig[:, :n_lbus] * P_sig + P_mu
    total_P   = P_D.sum(axis=1, keepdims=True)
    p_frac    = P_D / (total_P + 1e-8)
    log_scale = np.clip(np.log((total_P + 1e-8) / (nominal_total + 1e-8)), -3, 3)
    return np.concatenate(
        [L_val_orig, p_frac, log_scale,
         np.ones((n, n_gen),   dtype=float),
         np.ones((n, n_gen),   dtype=float),
         np.ones((n, n_gen),   dtype=float),
         np.ones((n, n_trafo), dtype=float)],
        axis=1,
    ).astype(np.float32)


def _save_npz(path, L_val_ext, x_type, Lg_Max_val, y_list,
              P_mu=None, P_sig=None, Q_mu=None, Q_sig=None, nominal_total=None):
    # Lg_Max_val may be a list of scalars (one per KKT output) or a plain scalar
    lg = np.array([float(np.asarray(x).ravel()[0]) for x in Lg_Max_val] if hasattr(Lg_Max_val, "__iter__") else [float(Lg_Max_val)])
    kwargs = {
        "L_val_ext": L_val_ext,
        "x_type":    x_type,
        "Lg_Max":    lg,
    }
    if P_mu is not None:
        kwargs.update({"P_mu": P_mu, "P_sig": P_sig, "Q_mu": Q_mu, "Q_sig": Q_sig,
                       "nominal_total": np.array([float(nominal_total)])})
    kwargs.update({"y_{}".format(i): arr for i, arr in enumerate(y_list)})
    np.savez_compressed(path, **kwargs)


def _load_npz(path):
    d = np.load(path, allow_pickle=False)
    L_val_ext = d["L_val_ext"]
    x_type    = d["x_type"]
    Lg_Max    = list(d["Lg_Max"])      # list of floats so PinnLayer can index [0],[1],[2]
    n_y       = sum(1 for k in d.files if k.startswith("y_"))
    y_list    = [d["y_{}".format(i)] for i in range(n_y)]
    P_mu      = d["P_mu"]            if "P_mu"          in d.files else None
    P_sig     = d["P_sig"]           if "P_sig"         in d.files else None
    nominal_total = float(d["nominal_total"][0]) if "nominal_total" in d.files else None
    return L_val_ext, x_type, Lg_Max, y_list, P_mu, P_sig, nominal_total


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--gen_data", action="store_true", default=False,
                        help="Build/rebuild nominal and targeted-stressed extended datasets.")
    parser.add_argument("--build_workers", type=int, default=-1,
                        help="-1=auto, 0/1=serial, >1=parallel norm-stats workers")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Epochs trained on nominal only before stressed samples appear.")
    parser.add_argument("--ramp_epochs", type=int, default=5,
                        help="Epochs over which stressed fraction ramps from 0 to 100%%.")
    parser.add_argument("--init_model_dir", type=str,
                        default="src/baselines/PINN/trained_model_ext_input/{num_bus}",
                        help="Dir of an extended-input model to warm-start from.")
    parser.add_argument("--save_model_dir", type=str,
                        default="src/baselines/PINN/trained_model_ext_input_augmented/{num_bus}")
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    num_bus = args.case[4:]
    ncpu = os.cpu_count() or 1
    if args.build_workers == -1:
        args.build_workers = max(1, min(8, ncpu - 1))

    init_model_dir = args.init_model_dir.format(num_bus=num_bus)
    save_model_dir = args.save_model_dir.format(num_bus=num_bus)

    dataset_dir   = _dataset_dir(num_bus)
    nom_path      = os.path.join(dataset_dir, "ext_input_dataset.npz")       # shared with train_ext_input
    stressed_path = os.path.join(dataset_dir, "ext_input_aug_stressed.npz")

    rng = np.random.default_rng(args.seed)

    base_net, _ = util.get_base_net(args.case, "in_distribution")
    simulation_parameters = create_example_parameters(base_net)
    n_lbus = simulation_parameters["general"]["n_lbus"]
    l_bus  = simulation_parameters["general"]["l_bus"]

    slack = int(base_net.ext_grid.bus.iloc[0])
    active_gen_idx = [
        i for i, (m, b) in enumerate(zip(
            base_net.gen.in_service.values.astype(bool), base_net.gen.bus.values))
        if m and int(b) != slack
    ]
    n_gen   = len(active_gen_idx)
    n_trafo = len(base_net.trafo.index)

    # ------------------------------------------------------------------
    # Build or load datasets
    # ------------------------------------------------------------------
    if args.gen_data:
        # --- Nominal (same as train_ext_input.py --gen_data) ---
        print("Build workers: {}".format(args.build_workers))
        print("Computing normalization stats from nominal scenarios...")
        P_mu, P_sig, Q_mu, Q_sig, nominal_total = _build_norm_stats(
            args.case, l_bus, args.build_workers)

        x_training_orig, y_training_nom, Lg_Max_nom_raw = create_data(base_net, simulation_parameters)
        L_val_orig_nom, x_type_nom = x_training_orig
        L_val_ext_nom = _make_extended_inputs(
            L_val_orig_nom, P_mu, P_sig, nominal_total, n_lbus, n_gen, n_trafo)
        Lg_Max_nom = [float(np.asarray(x).ravel()[0]) for x in Lg_Max_nom_raw]

        os.makedirs(dataset_dir, exist_ok=True)
        _save_npz(nom_path, L_val_ext_nom, x_type_nom, Lg_Max_nom, y_training_nom,
                  P_mu=P_mu, P_sig=P_sig, Q_mu=Q_mu, Q_sig=Q_sig, nominal_total=nominal_total)
        print("Saved nominal -> {}  ({} samples, in_dim={})".format(
            nom_path, L_val_ext_nom.shape[0], L_val_ext_nom.shape[1]))

        # --- Targeted stressed (TARGET_GROUPS only via CSV-swap trick) ---
        print("Building TARGET_GROUPS stressed cache: {}".format(TARGET_GROUPS))
        _build_target_stress_cache(args.case)

        print("Loading stressed arrays from targeted cache...")
        x_str_raw, y_str_raw, Lg_Max_str_raw = _load_target_stress_arrays(
            base_net, simulation_parameters, args.case)
        L_val_orig_str = x_str_raw[0]
        x_type_str     = x_str_raw[1]

        L_val_ext_str = _make_extended_inputs(
            L_val_orig_str, P_mu, P_sig, nominal_total, n_lbus, n_gen, n_trafo)
        Lg_Max_str = [float(np.asarray(x).ravel()[0]) for x in Lg_Max_str_raw]

        _save_npz(stressed_path, L_val_ext_str, x_type_str, Lg_Max_str, y_str_raw)
        print("Saved stressed -> {}  ({} samples, in_dim={})".format(
            stressed_path, L_val_ext_str.shape[0], L_val_ext_str.shape[1]))

    else:
        for p in (nom_path, stressed_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "Dataset not found: {}\nRun with --gen_data first.".format(p))

        L_val_ext_nom, x_type_nom, Lg_Max_nom, y_training_nom, P_mu, P_sig, nominal_total = \
            _load_npz(nom_path)
        L_val_ext_str, x_type_str, Lg_Max_str, y_str_raw, _, _, _ = _load_npz(stressed_path)

        print("Loaded nominal  from {}  ({} samples, in_dim={})".format(
            nom_path, L_val_ext_nom.shape[0], L_val_ext_nom.shape[1]))
        print("Loaded stressed from {}  ({} samples, in_dim={})".format(
            stressed_path, L_val_ext_str.shape[0], L_val_ext_str.shape[1]))

    # ------------------------------------------------------------------
    # Prepare list-of-arrays format for sample_epoch_dataset_full_union
    # ------------------------------------------------------------------
    x_nom = [L_val_ext_nom, x_type_nom]
    y_nom = list(y_training_nom)

    x_str = [L_val_ext_str, x_type_str]
    y_str = list(y_str_raw)

    n_nom_total = _nrows(x_nom)
    n_str_total = _nrows(x_str)
    print("Nominal: {}  Stressed: {}  Extended in_dim: {}".format(
        n_nom_total, n_str_total, L_val_ext_nom.shape[1]))

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    Lg_Max_combined = [max(a, b) for a, b in zip(Lg_Max_nom, Lg_Max_str)]
    simulation_parameters.update({"Lg_Max": Lg_Max_combined})

    if not args.train_from_scratch and os.path.exists(init_model_dir):
        # Load the full saved PinnModelExt directly — already built, correct concrete
        # function traces, no dummy-call interference.
        model = keras.models.load_model(init_model_dir)
        print("Warm-started from: {}".format(init_model_dir))
    else:
        # Fresh model — Keras will build it on the first model.fit call.
        model = PinnModelExt(0.05, 0.05, simulation_parameters)
        if not args.train_from_scratch:
            print("WARNING: init_model_dir not found ({}), training from scratch.".format(init_model_dir))
        else:
            print("Training from scratch.")

    # ------------------------------------------------------------------
    # Curriculum training loop
    # ------------------------------------------------------------------
    print("Starting extended-input PINN curriculum training")
    print("  epochs={}  warmup={}  ramp={}".format(
        args.epochs, args.warmup_epochs, args.ramp_epochs))

    for ep in range(args.epochs):
        n_stress_use = get_epoch_stress_count(ep, args.warmup_epochs, args.ramp_epochs, n_str_total)
        x_ep, y_ep, n_nom, n_stress, n_stress_total_ep = sample_epoch_dataset_full_union(
            x_nom, y_nom, x_str, y_str, n_stress_use, rng)
        stress_frac = float(n_stress) / float(n_stress_total_ep) if n_stress_total_ep > 0 else 0.0

        print("[epoch {:03d}/{:03d}] stress_frac={:.4f}  nominal={}  stressed={}  total={}".format(
            ep + 1, args.epochs, stress_frac, n_nom, n_stress, _nrows(x_ep)))

        model.fit(
            [x_ep[0], x_ep[1]],   # [L_val_ext, x_type]
            y_ep,
            epochs=1,
            batch_size=args.batch_size,
            verbose=0,
            shuffle=False,         # already shuffled by sample_epoch_dataset_full_union
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(save_model_dir, exist_ok=True)
    model.save(save_model_dir)
    print("Saved -> {}".format(save_model_dir))


if __name__ == "__main__":
    main()
