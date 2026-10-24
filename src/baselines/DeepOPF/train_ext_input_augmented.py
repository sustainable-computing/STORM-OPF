import argparse
import gzip
import io
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch

from src import util
from src.baselines.DeepOPF.train_augmented import (
    train_penalty_model_curriculum,
    set_runtime_threads,
    auto_main_cpu_threads,
    auto_penalty_workers,
)

from src.baselines.DeepOPF.train_ext_input import (
    _load_paths, _build_meta, _build_samples,
    _adopt_nonext_input_stats, _extend_nominal_from_saved,
)

ALL_GROUPS = [
    "in_distribution", "geo_shift", "extreme_weather",
    "load_extreme_high", "load_extreme_low",
    "fuel_price_up", "fuel_price_down",
    "nk_contingencies", "islanded_operation", "reactive_scarcity",
]
TARGET_GROUPS = ["geo_shift", "extreme_weather", "load_extreme_high", "load_extreme_low"]

_SAMPLE_META = None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Epochs to train on nominal data only before introducing stressed samples.")
    parser.add_argument("--ramp_epochs", type=int, default=5,
                        help="Epochs over which stressed fraction linearly ramps from 0 to 100%.")
    parser.add_argument("--batch", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--gen_data", action="store_true", default=False,
                        help="Build dataset from scenario files and save. "
                             "If False (default), load the previously saved datasets.")
    parser.add_argument("--build_workers", type=int, default=-1,
                        help="-1=auto, 0/1=serial, >1=parallel data-loading workers")
    parser.add_argument("--penalty_workers", type=int, default=-1,
                        help="-1=auto, 0/1=serial, >1=parallel PF penalty workers during training")

    parser.add_argument("--init_model_path", type=str, default="src/baselines/DeepOPF/model/{case}/trained_model_ext_input.pth",
                        help="Path to an extended-input model checkpoint to warm-start from. "
                             "Leave empty to train from scratch.")
    parser.add_argument("--train_from_scratch", action="store_true", default=False,
                        help="Ignore --init_model_path and train from random init.")
    parser.add_argument("--strict_load", action="store_true", default=False,
                        help="Require the init checkpoint to match the model architecture exactly.")

    parser.add_argument("--cpu_threads", type=int, default=0,
                        help="Main-process BLAS/PyTorch threads. 0=auto.")
    parser.add_argument("--interop_threads", type=int, default=1)
    args = parser.parse_args()

    ncpu = os.cpu_count() or 1
    if args.build_workers == -1:
        args.build_workers = max(1, min(8, ncpu - 1))
    if args.penalty_workers == -1:
        args.penalty_workers = auto_penalty_workers()
    if args.cpu_threads <= 0:
        args.cpu_threads = auto_main_cpu_threads()

    set_runtime_threads(cpu_threads=args.cpu_threads, interop_threads=args.interop_threads)

    dataset_dir = os.path.join("src", "baselines", "DeepOPF", "datasets", args.case)
    nom_path = os.path.join(dataset_dir, "ext_input_aug_nominal.pt")
    str_path = os.path.join(dataset_dir, "ext_input_aug_stressed.pt")

    base_net, _ = util.get_base_net(args.case, "in_distribution")

    if args.gen_data:
        print("Build workers: {}  Penalty workers: {}".format(args.build_workers, args.penalty_workers))
        print("Building meta from nominal scenarios ({})...".format(args.case))
        meta = _build_meta(args.case, args.build_workers)
        nd = _adopt_nonext_input_stats(meta, args.case)

        # Nominal: in_distribution (used as the base curriculum dataset).
        # Reuse the saved non-extended dataset when available (no net reads).
        if nd is not None:
            print("Reusing saved non-extended dataset for NOMINAL (no net reads)...")
            X_nom, Y_nom = _extend_nominal_from_saved(nd, meta)
        else:
            print("Building NOMINAL (in_distribution) extended dataset...")
            X_nom, Y_nom = _build_samples(_load_paths(args.case, "in_distribution"), meta, args.build_workers)
        print("  nominal: {} samples  in_dim={}".format(len(X_nom), X_nom.shape[1]))

        # Stressed: TARGET_GROUPS combined
        print("Building STRESSED (target groups) extended dataset...")
        all_X, all_Y = [], []
        for g in TARGET_GROUPS:
            X_g, Y_g = _build_samples(_load_paths(args.case, g), meta, args.build_workers)
            print("  {}: {} samples".format(g, len(X_g)))
            all_X.append(X_g)
            all_Y.append(Y_g)
        X_str = torch.cat(all_X)
        Y_str = torch.cat(all_Y)
        print("  stressed total: {} samples".format(len(X_str)))

        os.makedirs(dataset_dir, exist_ok=True)
        torch.save({"X": X_nom, "Y": Y_nom, "meta": meta, "groups": ["in_distribution"]}, nom_path)
        torch.save({"X": X_str, "Y": Y_str, "meta": meta, "groups": TARGET_GROUPS}, str_path)
        print("Saved nominal  -> {}".format(nom_path))
        print("Saved stressed -> {}".format(str_path))
    else:
        for p in (nom_path, str_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "Dataset not found: {}\nRun with --gen_data first.".format(p))
        loaded_nom = torch.load(nom_path, map_location="cpu")
        X_nom = loaded_nom["X"].float()
        Y_nom = loaded_nom["Y"].float()
        meta  = loaded_nom["meta"]

        loaded_str = torch.load(str_path, map_location="cpu")
        X_str = loaded_str["X"].float()
        Y_str = loaded_str["Y"].float()
        print("Loaded nominal  from {} ({} samples)".format(nom_path, len(X_nom)))
        print("Loaded stressed from {} ({} samples)".format(str_path, len(X_str)))

    in_dim = X_nom.shape[1]
    print("Extended in_dim = {}  (nominal={} stressed={} samples)".format(
        in_dim, len(X_nom), len(X_str)))

    init_model_path = ""
    if not args.train_from_scratch:
        init_model_path = args.init_model_path.format(case=args.case)

    model = train_penalty_model_curriculum(
        base_net=base_net,
        X_nom=X_nom,
        Y_nom=Y_nom,
        X_str=X_str,
        Y_str=Y_str,
        meta=meta,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        w1=args.w1,
        w2=args.w2,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        init_model_path=init_model_path,
        strict_load=args.strict_load,
        cpu_threads=args.cpu_threads,
        interop_threads=args.interop_threads,
        penalty_workers=args.penalty_workers,
    )

    model_dir = os.path.join("src", "baselines", "DeepOPF", "model", args.case)
    os.makedirs(model_dir, exist_ok=True)
    ckpt = os.path.join(model_dir, "trained_model_ext_input_augmented.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "meta": meta,
        "in_dim": in_dim,
        "groups": ["in_distribution"] + TARGET_GROUPS,
        "model_class": "DeepOPFPenaltyNet",
    }, ckpt)
    print("Saved model -> {}".format(ckpt))


if __name__ == "__main__":
    mp.freeze_support()
    main()
