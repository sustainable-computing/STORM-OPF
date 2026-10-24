import os
import copy
import json
import pickle
import shutil
import tempfile
import argparse
import contextlib
import random
import tensorflow as tf

import numpy as np
import pandas as pd
from tensorflow import keras
from src import util
from src.baselines.PINN.create_example_parameters import create_example_parameters
from src.baselines.PINN.create_data import create_data
from src.baselines.PINN.PinnModel import PinnModel
from src.baselines.PINN import generate_data

def _set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def _dataset_dir(num_bus: str) -> str:
    return os.path.join("src", "baselines", "PINN", "datasets", str(num_bus))


def _test_output_dir(num_bus: str) -> str:
    return os.path.join("src", "baselines", "PINN", "test_output", str(num_bus))



def _train_paths(num_bus: str):
    d = _dataset_dir(num_bus)
    return {
        "in": os.path.join(d, "NN_input_train.csv"),
        "pq": os.path.join(d, "NN_output_PQ_train.csv"),
        "v": os.path.join(d, "NN_output_V_train.csv"),
        "stats_in": os.path.join(d, "norm_stats.json"),
        "stats_pq": os.path.join(d, "norm_stats_pq.json"),
        "stats_v": os.path.join(d, "norm_stats_v.json"),
    }


def _stress_cache_paths(num_bus: str):
    d = _dataset_dir(num_bus)
    return {
        "in": os.path.join(d, "NN_input_train_stressed_for_augmented.csv"),
        "pq": os.path.join(d, "NN_output_PQ_train_stressed_for_augmented.csv"),
        "v": os.path.join(d, "NN_output_V_train_stressed_for_augmented.csv"),
        "meta": os.path.join(d, "training_stressed_for_augmented_meta.json"),
    }


def _apply_existing_stats(df: pd.DataFrame, stats_json_path: str) -> pd.DataFrame:
    return generate_data._apply_norm(df, stats_json_path)


@contextlib.contextmanager
def _temporary_train_csv_swap(num_bus: str, src_paths: dict):
    base = _train_paths(num_bus)
    out_dir = _test_output_dir(num_bus)
    os.makedirs(out_dir, exist_ok=True)
    debug_paths = {
        "features": os.path.join(out_dir, "features_train.csv"),
        "labels": os.path.join(out_dir, "labels_train.csv"),
    }

    with tempfile.TemporaryDirectory(prefix="pinn_aug_swap_") as tmpdir:
        backups = {}
        for key in ("in", "pq", "v"):
            backups[key] = os.path.join(tmpdir, os.path.basename(base[key]))
            if not os.path.exists(base[key]):
                raise FileNotFoundError(f"Missing baseline training CSV: {base[key]}")
            shutil.copy2(base[key], backups[key])
            shutil.copy2(src_paths[key], base[key])

        debug_backups = {}
        for key, path in debug_paths.items():
            debug_backups[key] = os.path.join(tmpdir, os.path.basename(path))
            if os.path.exists(path):
                shutil.copy2(path, debug_backups[key])
            else:
                open(debug_backups[key], "wb").close()

        try:
            yield
        finally:
            for key in ("in", "pq", "v"):
                shutil.copy2(backups[key], base[key])
            for key, path in debug_paths.items():
                if os.path.exists(debug_backups[key]) and os.path.getsize(debug_backups[key]) > 0:
                    shutil.copy2(debug_backups[key], path)
                elif os.path.exists(path):
                    os.remove(path)


def _build_stress_raw_dataframes(case: str):
    scen_root = os.path.join("src", "scenarios", case, "training_stressed")
    if not os.path.isdir(scen_root):
        raise FileNotFoundError(f"Missing stressed training directory: {scen_root}")

    stress_case_dirs = [
        os.path.join(scen_root, d)
        for d in sorted(os.listdir(scen_root))
        if os.path.isdir(os.path.join(scen_root, d))
    ]
    if len(stress_case_dirs) == 0:
        raise RuntimeError(f"No stress-case folders found under: {scen_root}")

    paths = []
    group_counts_manifest_ok = {}

    for group_dir in stress_case_dirs:
        group_name = os.path.basename(group_dir)
        manifest_path = os.path.join(group_dir, "manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing stressed training manifest: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        if "status" not in manifest.columns or "file" not in manifest.columns:
            raise ValueError(f"Manifest must contain columns ['status', 'file']: {manifest_path}")

        manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
        group_counts_manifest_ok[group_name] = int(len(manifest))

        for rel_file in manifest["file"].tolist():
            path = os.path.join(group_dir, str(rel_file))
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scenario file listed in manifest not found: {path}")
            paths.append((group_name, path))

    if len(paths) == 0:
        raise RuntimeError(f"No stressed scenario files found under: {scen_root}")

    rows_in, rows_pq, rows_v = [], [], []
    kept_groups = []
    skipped = 0

    for group_name, path in paths:
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
        kept_groups.append(group_name)

    if not rows_in:
        raise RuntimeError("No feasible stressed training samples were collected.")

    df_in = pd.DataFrame(rows_in)
    df_pq = pd.DataFrame(rows_pq)
    df_v = pd.DataFrame(rows_v)

    df_in = df_in.reindex(generate_data._natural_sort(df_in.columns), axis=1)
    df_pq = df_pq.reindex(generate_data._natural_sort(df_pq.columns), axis=1)
    df_v = df_v.reindex(generate_data._natural_sort(df_v.columns), axis=1)

    perm = np.random.permutation(len(df_in))
    df_in = df_in.iloc[perm].reset_index(drop=True)
    df_pq = df_pq.iloc[perm].reset_index(drop=True)
    df_v = df_v.iloc[perm].reset_index(drop=True)
    kept_groups = [kept_groups[i] for i in perm.tolist()]

    kept_group_counts = {}
    for g in kept_groups:
        kept_group_counts[g] = kept_group_counts.get(g, 0) + 1

    meta = {
        "n_total_manifest_ok": int(len(paths)),
        "n_kept": int(len(df_in)),
        "n_skipped": int(skipped),
        "stress_groups": sorted(group_counts_manifest_ok.keys()),
        "manifest_ok_per_group": group_counts_manifest_ok,
        "kept_per_group": kept_group_counts,
    }

    print(f"[stress-cache] loaded stressed samples from {len(group_counts_manifest_ok)} folders")
    print(f"[stress-cache] manifest-ok per folder: {group_counts_manifest_ok}")
    print(f"[stress-cache] kept per folder after OPF filtering: {kept_group_counts}")
    print(f"[stress-cache] total kept={len(df_in)} skipped={skipped}")

    return df_in, df_pq, df_v, meta


def generate_stress_training_data_for_augmented(case: str):
    num_bus = case[4:]
    out_dir = _dataset_dir(num_bus)
    os.makedirs(out_dir, exist_ok=True)

    base = _train_paths(num_bus)
    stress = _stress_cache_paths(num_bus)

    for p in (base["in"], base["pq"], base["v"], base["stats_in"], base["stats_pq"], base["stats_v"]):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required baseline training artifact not found: {p}")

    df_stress_in_raw, df_stress_pq_raw, df_stress_v_raw, meta = _build_stress_raw_dataframes(case)

    df_stress_in_n = _apply_existing_stats(df_stress_in_raw, base["stats_in"])
    df_stress_pq_n = _apply_existing_stats(df_stress_pq_raw, base["stats_pq"])
    df_stress_v_n = _apply_existing_stats(df_stress_v_raw, base["stats_v"])

    df_stress_in_n.to_csv(stress["in"], index=False)
    df_stress_pq_n.to_csv(stress["pq"], index=False)
    df_stress_v_n.to_csv(stress["v"], index=False)
    with open(stress["meta"], "w") as f:
        json.dump(meta, f, indent=2)


def _assert_stress_cache_exists(case: str):
    num_bus = case[4:]
    stress = _stress_cache_paths(num_bus)
    for p in (stress["in"], stress["pq"], stress["v"]):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing stressed training cache: {p}. Run with --gen_data once first.")


def _to_numpy_list(seq):
    return [np.asarray(x) for x in seq]


def _load_nominal_training_arrays(net, simulation_parameters):
    x_training, y_training, Lg_Max = create_data(net, simulation_parameters=copy.deepcopy(simulation_parameters))
    return _to_numpy_list(x_training), _to_numpy_list(y_training), [np.asarray(x) for x in Lg_Max]


def _load_stress_training_arrays(net, simulation_parameters, case: str):
    num_bus = case[4:]
    stress = _stress_cache_paths(num_bus)
    with _temporary_train_csv_swap(num_bus, stress):
        x_training, y_training, Lg_Max = create_data(net, simulation_parameters=copy.deepcopy(simulation_parameters))
    return _to_numpy_list(x_training), _to_numpy_list(y_training), [np.asarray(x) for x in Lg_Max]


def _nrows(arr_list):
    return int(arr_list[0].shape[0])


def _slice_list(arr_list, idx):
    return [arr[idx] for arr in arr_list]


def _concat_lists(a_list, b_list):
    return [np.concatenate([a, b], axis=0) for a, b in zip(a_list, b_list)]


def _permute_list(arr_list, perm):
    return [arr[perm] for arr in arr_list]


def _count_data_and_collocation(x_list):
    x_type = x_list[1]
    n_data = int(np.sum(x_type.reshape(-1) > 0.5))
    n_total = int(x_type.shape[0])
    return n_data, n_total - n_data


def _combine_lg_max(lg_nom, lg_str):
    return [np.maximum(a, b) for a, b in zip(lg_nom, lg_str)]


def _load_initial_model_weights(model, init_model_dir: str, train_from_scratch: bool):
    if train_from_scratch:
        print("Training augmented PINN from scratch")
        return model

    if not os.path.exists(init_model_dir):
        raise FileNotFoundError(f"Baseline model directory not found: {init_model_dir}")

    loaded_model = keras.models.load_model(init_model_dir)
    model.set_weights(loaded_model.get_weights())
    print(f"Loaded BASELINE weights from: {init_model_dir}")
    return model


def get_epoch_stress_count(epoch: int, warmup_epochs: int, ramp_epochs: int, n_stress_total: int) -> int:
    if epoch < warmup_epochs:
        return 0
    if ramp_epochs <= 0:
        return n_stress_total
    prog = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return int(round(prog * n_stress_total))


def sample_epoch_dataset_full_union(x_nom, y_nom, x_str, y_str, n_stress_use: int, rng):
    n_nom = _nrows(x_nom)
    if n_stress_use <= 0:
        perm = rng.permutation(n_nom)
        return _permute_list(x_nom, perm), _permute_list(y_nom, perm), n_nom, 0, 0

    n_stress_total = _nrows(x_str)
    n_stress_use = min(int(n_stress_use), n_stress_total)
    idx_str = rng.permutation(n_stress_total)[:n_stress_use]

    x_str_use = _slice_list(x_str, idx_str)
    y_str_use = _slice_list(y_str, idx_str)

    x_epoch = _concat_lists(x_nom, x_str_use)
    y_epoch = _concat_lists(y_nom, y_str_use)

    perm = rng.permutation(_nrows(x_epoch))
    x_epoch = _permute_list(x_epoch, perm)
    y_epoch = _permute_list(y_epoch, perm)
    return x_epoch, y_epoch, n_nom, n_stress_use, n_stress_total


def train_model_augmented(
    net,
    gen_data,
    case,
    epochs,
    batch_size,
    warmup_epochs,
    ramp_epochs,
    init_model_dir,
    save_model_dir,
    train_from_scratch,
    seed,
):
    
    _set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    simulation_parameters = create_example_parameters(net)
    num_bus = case[4:]

    if gen_data:
        generate_stress_training_data_for_augmented(case)
    else:
        _assert_stress_cache_exists(case)

    x_str, y_str, Lg_Max_str = _load_stress_training_arrays(net, simulation_parameters, case)
    x_nom, y_nom, Lg_Max_nom = _load_nominal_training_arrays(net, simulation_parameters)

    n_nom_total = _nrows(x_nom)
    n_str_total = _nrows(x_str)
    n_nom_data, n_nom_colloc = _count_data_and_collocation(x_nom)
    n_str_data, n_str_colloc = _count_data_and_collocation(x_str)

    simulation_parameters["data_creation"]["n_data_points"] = n_nom_data + n_str_data
    simulation_parameters["data_creation"]["n_collocation"] = n_nom_colloc + n_str_colloc
    simulation_parameters.update({"Lg_Max": _combine_lg_max(Lg_Max_nom, Lg_Max_str)})

    lw1 = 0.05
    lw2 = 0.05
    model = PinnModel(lw1, lw2, simulation_parameters=simulation_parameters)
    model = _load_initial_model_weights(
        model=model,
        init_model_dir=init_model_dir,
        train_from_scratch=train_from_scratch,
    )

    np.set_printoptions(precision=3)
    print("Starting augmented PINN training")
    print(f"Nominal samples: {n_nom_total}")
    print(f"Stressed samples: {n_str_total}")

    for ep in range(epochs):
        n_stress_use = get_epoch_stress_count(ep, warmup_epochs, ramp_epochs, n_str_total)
        x_ep, y_ep, n_nominal, n_stress, n_stress_total_epoch = sample_epoch_dataset_full_union(
            x_nom, y_nom, x_str, y_str, n_stress_use, rng
        )
        stress_frac = float(n_stress) / float(n_stress_total_epoch) if n_stress_total_epoch > 0 else 0.0

        print(
            f"[epoch {ep + 1:03d}/{epochs:03d}] "
            f"stress_frac={stress_frac:.4f} "
            f"nominal={n_nominal} stressed={n_stress} total={_nrows(x_ep)}"
        )

        model.fit(
            x_ep,
            y_ep,
            epochs=1,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,   #no shuffling again to avoid randomness
        )

    os.makedirs(save_model_dir, exist_ok=True)
    model.save(save_model_dir)

    sim_param_path = os.path.join(
        "src", "baselines", "PINN", "datasets", num_bus, "simulation_parameters_augmented.pkl"
    )
    with open(sim_param_path, "wb") as f:
        pickle.dump(simulation_parameters, f)

    print(f"Saved augmented PINN model to: {save_model_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--gen_data", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--ramp_epochs", type=int, default=5)
    parser.add_argument(
        "--init_model_dir",
        type=str,
        default="src/baselines/PINN/trained_model/{num_bus}",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="src/baselines/PINN/trained_model_augmented/{num_bus}",
    )
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    num_bus = args.case[4:]
    init_model_dir = args.init_model_dir.format(num_bus=num_bus)
    save_model_dir = args.save_model_dir.format(num_bus=num_bus)

    base_net, _ = util.get_base_net(args.case, "in_distribution")
    train_model_augmented(
        net=base_net,
        gen_data=args.gen_data,
        case=args.case,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        init_model_dir=init_model_dir,
        save_model_dir=save_model_dir,
        train_from_scratch=args.train_from_scratch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()