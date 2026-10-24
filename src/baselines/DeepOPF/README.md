# Train model using pre-saved training data

`python3 -m src.baselines.DeepOPF.train_model --case case14`

Note: use `--gen_data` flag to create training dataset again (i.e. don't use the pre-saved train data) before training. 


# Train augmented model using stressed training data


If the stressed dataset is already created, you can train again without regenerating it:

`python3 -m src.baselines.DeepOPF.train_augmented --case case14`

Otherwise, use the `gen_data` flag to create the stressed data. 


Note that, by default, the augmented model is initialized from the existing base model and saved as:

`src/baselines/DeepOPF/model/{case}/trained_model_augmented.pth`

To train the augmented model from scratch instead of loading the base model, use:

`python3 -m src.baselines.DeepOPF.train_augmented --case case14 --train_from_scratch`




# Test model on test datasets from all scenarios 
Simply run the `.sh` file:

`./src/baselines/DeepOPF/test.sh`

# Test model one one particular scenario

`python -m src.baselines.DeepOPF.test_model --which_data 'in_distribution'` 

