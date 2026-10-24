 # Train model  
 
`python3 -m src.baselines.PINN.train_model --case case14`

Note: use `gen_data` flag to generate training data before training.



# Test on all cases listed in the .sh file:  

`./src/baselines/PINN/test.sh --case case14`

Note: use `gen_data` flag to generate the test data, otherwise it'll try using pre-saved tets data.