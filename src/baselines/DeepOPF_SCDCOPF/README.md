## Generate train data
python -m src.baselines.DeepOPF_SCDCOPF.make_dataset \
  --case case14 \
  --purpose training \    
  --max_samples_train 10000\
  --num_proc 20 \
  --math_threads_per_proc 5 \
  --checkpoint_every 100 \
  --progress_every 20

Note: change purpose to generate `stress_train` data.

## Generate test data
python -m src.baselines.DeepOPF_SCDCOPF.make_dataset \
  --case case14 \
  --purpose test \
  --which_data in_distribution \
  --num_proc 16 \
  --math_threads_per_proc 1 \
  --max_samples_test 1000 \
  --checkpoint_every 100 \
  --progress_every 20

## Train Model
python -m src.baselines.DeepOPF_SCDCOPF.train_model --case case14 --which_model baseline

Note: use `--which_model augmented` to train a stress-data-augmented model.

## Test (without projection)
./src/baselines/DeepOPF_SCDCOPF/test.sh --which_model baseline --no_projection

## Test (with projection)
./src/baselines/DeepOPF_SCDCOPF/test.sh --which_model baseline 