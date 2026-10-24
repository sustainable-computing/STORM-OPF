# Three-gap experiments (DeepOPF)

Scaffolding for the "restoration under distribution shift" study, instantiated
for DeepOPF. All heavy lifting (feature builder, network, training loop,
denorm/apply adapters, SHIELD operator) is **reused** from existing files;
this folder only wires them into the experiment matrix and adds parallel
evaluation. No existing file is modified.

## The matrix

| Axis | Values | Flag |
|---|---|---|
| Feature mode (gap 1) | `loads`, `extended` | `--mode` |
| Training regime | `nominal`, `augmented`, `leave_one_out` | `--regime` (+`--held_out`) |
| Restorer (gap 2) | `none`, `shield` *(more to come)* | `--restorer` |
| Deferral | `none`, `pdipm` | `--fallback` |

- **Gap 1** — `loads` vs `extended` at the same regime. Under `nominal`
  the extended features are degenerate (ratios ≈ 1), so the difference shows
  up mainly under `augmented` / `leave_one_out`.
- **Gap 2** — fix the model, vary `--restorer`. `none` is the constraint-baked
  raw prediction; `shield` is constraint-conditioned. Future operators
  (FRMNet-style learned repair, DC/L1-QP projection, DC3 gradient correction,
  gauge maps) drop into `RESTORERS` in `restorers.py` with the same signature.
  FSNet/DC3 *end-to-end* methods are separate trained models, not restorers.
- **Gap 3** — `augmented` regime, with/without restorer, to size the residual
  approximation gap.
- **leave_one_out** is the clincher: augment with 3 hard groups, test on the
  held-out 4th. Shows learned mechanisms generalize only to *seen* stressors,
  while shield is indifferent to which stressor moved the limits.

## Run

```bash
# gap-1 pair (nominal)
python -m src.baselines.DeepOPF.gaps.train --case case14 --mode loads    --regime nominal --gen_data
python -m src.baselines.DeepOPF.gaps.train --case case14 --mode extended --regime nominal

# gap-2: same model, two restorers, on a constraint-shift scenario
python -m src.baselines.DeepOPF.gaps.evaluate --case case14 --mode extended --regime nominal \
    --which_data extreme_weather --restorer none  --gen_data
python -m src.baselines.DeepOPF.gaps.evaluate --case case14 --mode extended --regime nominal \
    --which_data extreme_weather --restorer shield --fallback pdipm

# gap-3 / leave-one-out
python -m src.baselines.DeepOPF.gaps.train --case case14 --mode extended --regime augmented --gen_data
python -m src.baselines.DeepOPF.gaps.train --case case14 --mode extended \
    --regime leave_one_out --held_out extreme_weather --gen_data
```

`--gen_data` builds and caches the dataset (train) or test-feature tensor
(evaluate); omit it on later runs to load the cache. Both train and evaluate
auto-pick worker counts; override with `--build_workers` / `--penalty_workers`
/ `--workers`.

## Artifacts

```
src/baselines/DeepOPF/datasets/<case>/gaps_train_<regime_tag>.pt   # extended train cache
src/baselines/DeepOPF/datasets/<case>/gaps_test_<scenario>.pt       # extended test cache
src/baselines/DeepOPF/model/<case>/gaps/<mode>_<regime_tag>.pth     # checkpoint
results/<case>/DeepOPF_gaps_<mode>_<regime_tag>_<restorer>_result_<scenario>.pkl
```

`<regime_tag>` is `nominal`, `augmented`, or `loo_<held_out>`.

Result pkls carry the same keys as the other DeepOPF results
(`feasibility_rate_%`, `costs`, `avg_cost`, `hard_feasible_flags`) plus
study-specific fields (`feasibility_lower_bound_%`, `projection_rate_%`,
`deferral_rate_%`, `final_feasibility_rate_%`, timing).

## Files

- `data.py` — regimes, dataset gen/load/cache, feature-mode slicing (parallel).
- `train.py` — train + tagged checkpoint (reuses `_train_model_with_workers`).
- `restorers.py` — restorer registry (`none`, `shield`).
- `evaluate.py` — parallel apply + restore + PF-verify + deferral; saves pkl.
