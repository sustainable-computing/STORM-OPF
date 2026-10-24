"""
Evaluate a trained gap-study DeepOPF model, optionally wrapped with a
restoration operator, on one test scenario.  Power flows run in parallel.

The full matrix for a scenario is (feature mode) x (regime) x (restorer):
  --mode loads|extended   --regime nominal|augmented|leave_one_out
  --restorer none|shield   [--fallback pdipm]

Reuses DeepOPF's denorm/apply adapters and the restorer registry; the per-
sample power-flow work mirrors ext_input_shield/test_shield for speed.

Example:
  python -m src.baselines.DeepOPF.gaps.evaluate --case case14 \
      --mode extended --regime nominal --which_data extreme_weather \
      --restorer shield --fallback pdipm --gen_data
"""
import argparse
import concurrent.futures
import copy
import math
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.DeepOPF.train_model import DeepOPFPenaltyNet
from src.baselines.DeepOPF.test_model import (
    deepopf_apply_prediction_to_net,
    deepopf_denorm_outputs,
)
from src.baselines.DeepOPF import vars as dvars
from src.baselines.DeepOPF.gaps.data import gen_or_load_test, slice_mode
from src.baselines.DeepOPF.gaps.restorers import RESTORERS
from src.baselines.DeepOPF.gaps.train import ckpt_path, model_tag


# -----------------------------------------------------------------------
# Per-sample worker (separate process; fork inherits module-level imports)
# -----------------------------------------------------------------------

def _worker(task):
    path, x_np, y_np, meta, restorer, use_fallback = task
    import pandapower as _pp
    from src import util as _util
    from src.global_evaluation_AC import (
        evaluate_total_cost as _cost,
        get_orig_pq_bus_idx as _pq,
        is_feasible_by_hard_constraints as _hard,
    )
    from src.baselines.ext_input_shield.shield import pdipm_fallback as _fb

    out = {
        "feasible": False, "cost": float("nan"),
        "n_pf": 0, "projected": False, "vm_corrected": False,
        "deferred": False, "final_feasible": False, "final_cost": float("nan"),
        "restore_time_s": float("nan"), "fallback_time_s": 0.0,
    }
    try:
        net = copy.deepcopy(_util.load_net(path))
        pq_idx = _pq(net)
        x_t = torch.as_tensor(x_np, dtype=torch.float32)
        y_t = torch.as_tensor(y_np, dtype=torch.float32)
        net = deepopf_apply_prediction_to_net(
            net, meta, x_t, y_t, deepopf_denorm_outputs
        )
        t0 = time.perf_counter()
        feas, n_pf, proj, vmc = RESTORERS[restorer](net, meta, pq_idx)
        out["restore_time_s"] = time.perf_counter() - t0
        out.update(feasible=feas, n_pf=n_pf, projected=proj, vm_corrected=vmc)
        if bool(getattr(net, "converged", False)):
            out["cost"] = _cost(net)
    except Exception:
        pass

    out["final_feasible"] = out["feasible"]
    out["final_cost"] = out["cost"]
    if use_fallback and not out["feasible"]:
        out["deferred"] = True
        t0 = time.perf_counter()
        fb = _fb(path)
        out["fallback_time_s"] = time.perf_counter() - t0
        out["final_feasible"] = fb["feasible"]
        out["final_cost"] = fb["cost"]
    return out


def clopper_pearson_lower(successes, n, confidence=0.95):
    if n == 0:
        return float("nan")
    if successes == 0:
        return 0.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - confidence, successes, n - successes + 1))
    except Exception:
        p = successes / n
        z = 1.6448536269514722
        return max(0.0, p - z * math.sqrt(p * (1 - p) / n))


def _load_model(case, mode, regime, held_out, device):
    path = ckpt_path(case, mode, regime, held_out)
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["meta"]
    in_dim = int(ckpt["in_dim"])
    out_dim = len(meta["active_gen_idx"]) + 1 + len(meta["pv_bus_list"])
    model = DeepOPFPenaltyNet(in_dim, hidden=dvars.model_hidden, out_dim=out_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="case14")
    p.add_argument("--mode", choices=["loads", "extended"], default="extended")
    p.add_argument("--regime", choices=["nominal", "augmented", "leave_one_out"],
                   default="nominal")
    p.add_argument("--held_out", default=None)
    p.add_argument("--which_data", default="extreme_weather")
    p.add_argument("--restorer", choices=list(RESTORERS), default="none")
    p.add_argument("--fallback", choices=["none", "pdipm"], default="none")
    p.add_argument("--batch", type=int, default=3072)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gen_data", action="store_true",
                   help="Rebuild the test feature cache for this scenario")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--workers", type=int, default=-1)
    args = p.parse_args()

    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 1) - 1)
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    model, meta = _load_model(args.case, args.mode, args.regime, args.held_out, device)
    X, _, paths = gen_or_load_test(args.case, args.which_data, meta, n_workers, args.gen_data)
    X = slice_mode(X, args.mode, meta["n_bus"])
    if args.limit:
        X, paths = X[: args.limit], paths[: args.limit]

    # batched inference
    loader = DataLoader(TensorDataset(X), batch_size=args.batch, shuffle=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = torch.cat([model(xb.to(device)).cpu() for (xb,) in loader])
    infer_time = (time.perf_counter() - t0) / max(1, len(paths))

    tasks = [
        (paths[i], X[i].numpy(), preds[i].numpy(), meta, args.restorer,
         args.fallback == "pdipm")
        for i in range(len(paths))
    ]
    tag = model_tag(args.mode, args.regime, args.held_out)
    print(f"Eval | {tag} | restorer={args.restorer} | {args.which_data} | "
          f"n={len(paths)} workers={n_workers} fallback={args.fallback}")
    chunk = max(1, len(paths) // (n_workers * 4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        rows = list(pool.map(_worker, tasks, chunksize=chunk))

    n = len(rows)
    feas = [r["feasible"] for r in rows]
    final = [r["final_feasible"] for r in rows]
    costs = [r["cost"] if r["feasible"] else float("nan") for r in rows]
    final_costs = [r["final_cost"] if r["final_feasible"] else float("nan") for r in rows]
    n_def = sum(r["deferred"] for r in rows)

    result = {
        "n_samples": n,
        "model_tag": tag, "restorer": args.restorer, "which_data": args.which_data,
        "hard_feasible_flags": feas,
        "feasibility_rate_%": 100.0 * float(np.mean(feas)),
        "costs": costs,
        "avg_cost": float(np.nanmean(costs)) if any(feas) else float("nan"),
        "feasibility_lower_bound_%": 100.0 * clopper_pearson_lower(
            int(np.sum(feas)), n, args.confidence),
        "confidence": args.confidence,
        "projection_rate_%": 100.0 * float(np.mean([r["projected"] for r in rows])),
        "vm_correction_rate_%": 100.0 * float(np.mean([r["vm_corrected"] for r in rows])),
        "avg_pf_runs": float(np.mean([r["n_pf"] for r in rows])),
        "fallback": args.fallback,
        "deferral_rate_%": 100.0 * n_def / n,
        "final_feasibility_rate_%": 100.0 * float(np.mean(final)),
        "final_avg_cost": float(np.nanmean(final_costs)) if any(final) else float("nan"),
        "infer_time_per_sample_s": infer_time,
        "restore_time_per_sample_s": float(np.nanmean([r["restore_time_s"] for r in rows])),
    }

    print(f"  feasibility   : {result['feasibility_rate_%']:.1f}%  "
          f"(>= {result['feasibility_lower_bound_%']:.1f}% @ {int(args.confidence*100)}%)")
    print(f"  proj/vm/pf    : {result['projection_rate_%']:.1f}% / "
          f"{result['vm_correction_rate_%']:.1f}% / {result['avg_pf_runs']:.2f}")
    if args.fallback == "pdipm":
        print(f"  deferral      : {result['deferral_rate_%']:.1f}%  "
              f"deployed: {result['final_feasibility_rate_%']:.1f}%")

    result_dir = os.path.join("results_gap", args.case)
    os.makedirs(result_dir, exist_ok=True)
    out = os.path.join(
        result_dir,
        f"DeepOPF_gaps_{tag}_{args.restorer}_result_{args.which_data}.pkl",
    )
    with open(out, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
