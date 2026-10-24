"""
Train a DeepOPF predictor for the three-gap study.

Two feature modes  : loads | extended          (gap 1)
Three regimes       : nominal | augmented | leave_one_out   (training axis)

The network, the MSE + zeroth-order-penalty training loop, and all parallelism
are reused from train_ext_input._train_model_with_workers.  This file only
selects the data, slices the feature mode, and saves a tagged checkpoint.

Examples:
  # gap-1 pair, nominal protocol
  python -m src.baselines.DeepOPF.gaps.train --case case14 --mode loads    --regime nominal --gen_data
  python -m src.baselines.DeepOPF.gaps.train --case case14 --mode extended --regime nominal

  # gap-3, augmented
  python -m src.baselines.DeepOPF.gaps.train --case case14 --mode extended --regime augmented --gen_data

  # gap-2 proof, hold out extreme_weather
  python -m src.baselines.DeepOPF.gaps.train --case case14 --mode extended \
         --regime leave_one_out --held_out extreme_weather --gen_data
"""
import argparse
import os

import torch

from src import util
from src.baselines.DeepOPF.train_ext_input import _train_model_with_workers
from src.baselines.DeepOPF.gaps.data import (
    gen_or_load_train,
    regime_tag,
    slice_mode,
)


def model_tag(mode, regime, held_out=None):
    return f"{mode}_{regime_tag(regime, held_out)}"


def ckpt_path(case, mode, regime, held_out=None):
    return os.path.join(
        "src", "baselines", "DeepOPF", "model", case, "gaps",
        f"{model_tag(mode, regime, held_out)}.pth",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="case14")
    p.add_argument("--mode", choices=["loads", "extended"], default="extended")
    p.add_argument("--regime", choices=["nominal", "augmented", "leave_one_out"],
                   default="nominal")
    p.add_argument("--held_out", default=None,
                   help="Stress group to hold out (leave_one_out regime only)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=3072)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gen_data", action="store_true",
                   help="Rebuild the training dataset (else load cache)")
    p.add_argument("--build_workers", type=int, default=-1)
    p.add_argument("--penalty_workers", type=int, default=-1)
    args = p.parse_args()

    ncpu = os.cpu_count() or 1
    if args.build_workers == -1:
        args.build_workers = max(1, min(8, ncpu - 1))
    if args.penalty_workers == -1:
        args.penalty_workers = max(1, min(8, ncpu - 1))

    X, Y, meta = gen_or_load_train(
        args.case, args.regime, args.held_out, args.build_workers, args.gen_data
    )
    X = slice_mode(X, args.mode, meta["n_bus"])
    print(f"Training DeepOPF | mode={args.mode} regime={args.regime} "
          f"held_out={args.held_out} | in_dim={X.shape[1]} out_dim={Y.shape[1]}")

    base_net, _ = util.get_base_net(args.case, "in_distribution")
    model = _train_model_with_workers(
        base_net, X, Y, meta,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        device=args.device, w1=args.w1, w2=args.w2,
        penalty_workers=args.penalty_workers,
    )

    out = ckpt_path(args.case, args.mode, args.regime, args.held_out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "meta": meta,
        "mode": args.mode, "regime": args.regime, "held_out": args.held_out,
        "in_dim": X.shape[1], "model_class": "DeepOPFPenaltyNet",
    }, out)
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
