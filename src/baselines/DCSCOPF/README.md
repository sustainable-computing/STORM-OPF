## Evaluate iterative DC SC-OPF on all test scenarios (uses DC constraints)
`python3 -m src.baselines.DCSCOPF.evaluate_iterative_dcscopf --case case14 --num_workers 10 --verbose`


## Evaluate DC SC-OPF using AC constraints
`python3 -m src.baselines.DCSCOPF.evaluate_dcscopf_acConst --case case14 --which_data all --verbose`