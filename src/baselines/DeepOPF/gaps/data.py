"""
Data pipeline for the three-gap experiments (DeepOPF).

One cached tensor file per *training regime* holds the EXTENDED feature matrix
(loads + capacity/cost/trafo ratios) plus labels.  The loads-only feature set
is just the first 2*n_bus columns, so a single cached file serves both
feature modes — sliced on demand.  Nothing here is DeepOPF-specific except the
feature/label layout, which is reused verbatim from train_ext_input.

Regimes (which scenario groups enter the training set):
  nominal        in_distribution only            (primary protocol)
  augmented      in_distribution + 4 hard groups (gap-3 tool)
  leave_one_out  in_distribution + 3 hard groups (hold one out for gap-2 proof)
"""
import os

import torch

import numpy as np

from src.baselines.DeepOPF.train_ext_input import (
    TARGET_GROUPS,
    _build_meta,
    _build_one_sample,
    _build_one_sample_worker,
    _build_samples,
    _load_paths,
    _mp_context,
    _worker_init,
)

DATA_DIR = os.path.join("src", "baselines", "DeepOPF", "datasets")


def regime_groups(regime, held_out=None):
    """Scenario groups that make up the training set for a regime."""
    if regime == "nominal":
        return ["in_distribution"]
    if regime == "augmented":
        return ["in_distribution"] + list(TARGET_GROUPS)
    if regime == "leave_one_out":
        if held_out not in TARGET_GROUPS:
            raise ValueError(
                f"leave_one_out needs --held_out in {TARGET_GROUPS}, got {held_out!r}"
            )
        return ["in_distribution"] + [g for g in TARGET_GROUPS if g != held_out]
    raise ValueError(f"unknown regime {regime!r}")


def regime_tag(regime, held_out=None):
    return regime if regime != "leave_one_out" else f"loo_{held_out}"


def slice_mode(X, mode, n_bus):
    """Select the feature columns for a feature mode from the extended matrix."""
    if mode == "extended":
        return X
    if mode == "loads":
        return X[:, : 2 * n_bus].contiguous()
    raise ValueError(f"unknown feature mode {mode!r} (use loads|extended)")


def _dataset_path(case, regime, held_out):
    return os.path.join(DATA_DIR, case, f"gaps_train_{regime_tag(regime, held_out)}.pt")


def gen_or_load_train(case, regime, held_out, workers, regen):
    """
    Build (or load) the extended training dataset for a regime.
    Returns (X_extended, Y, meta).  Caller slices X for the feature mode.
    """
    path = _dataset_path(case, regime, held_out)
    if not regen and os.path.exists(path):
        d = torch.load(path, map_location="cpu")
        print(f"Loaded train dataset {path}  ({len(d['X'])} samples)")
        return d["X"].float(), d["Y"].float(), d["meta"]

    meta = _build_meta(case, workers)
    groups = regime_groups(regime, held_out)
    Xs, Ys = [], []
    for g in groups:
        X, Y = _build_samples(_load_paths(case, g), meta, workers)
        print(f"  {g}: {len(X)} samples")
        Xs.append(X)
        Ys.append(Y)
    X = torch.cat(Xs)
    Y = torch.cat(Ys)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"X": X, "Y": Y, "meta": meta, "groups": groups}, path)
    print(f"Saved train dataset -> {path}  ({len(X)} samples, in_dim={X.shape[1]})")
    return X.float(), Y.float(), meta


def _build_aligned(paths, meta, workers):
    """similar to _build_samples but keeps X/Y aligned with the surviving paths."""
    if workers and workers > 1:
        chunk = max(1, len(paths) // (workers * 4))
        with _mp_context().Pool(
            processes=workers, initializer=_worker_init, initargs=(meta,)
        ) as pool:
            rows = pool.map(_build_one_sample_worker, paths, chunksize=chunk)
    else:
        rows = [_build_one_sample(p, meta) for p in paths]

    kept_paths, Xs, Ys = [], [], []
    for p, row in zip(paths, rows):
        if row is None:
            continue
        kept_paths.append(p)
        Xs.append(row[0])
        Ys.append(row[1])
    if not Xs:
        raise RuntimeError(f"No valid test samples among {len(paths)} paths.")
    X = torch.tensor(np.vstack(Xs), dtype=torch.float32)
    Y = torch.tensor(np.vstack(Ys), dtype=torch.float32)
    return X, Y, kept_paths


def gen_or_load_test(case, which_data, meta, workers, regen):
    """
    Build (or load) the extended test feature matrix for one scenario.
    Cached independently of the model/regime (features depend only on meta).
    Returns (X_extended, Y, paths) with rows aligned to paths.
    """
    import pandas as pd

    path = os.path.join(DATA_DIR, case, f"gaps_test_{which_data}.pt")
    if not regen and os.path.exists(path):
        d = torch.load(path, map_location="cpu")
        print(f"Loaded test dataset {path}  ({len(d['X'])} samples)")
        return d["X"].float(), d["Y"].float(), d["paths"]

    scen_dir = os.path.join("src", "scenarios", case, "test", which_data)
    mf = pd.read_csv(os.path.join(scen_dir, "manifest.csv"))
    mf = mf[mf["status"] == "ok"].reset_index(drop=True)
    paths = [os.path.join(scen_dir, f) for f in mf["file"].tolist()]
    X, Y, kept = _build_aligned(paths, meta, workers)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"X": X, "Y": Y, "paths": kept}, path)
    print(f"Saved test dataset -> {path}  ({len(X)} samples)")
    return X.float(), Y.float(), kept
