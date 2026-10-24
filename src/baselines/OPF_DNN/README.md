### To create the environment
1. Open terminal from the folder where we have env_opf_dnn.yml & setup_julia_env.jl for OPF-DNN
2. conda env create -f env_opf_dnn.yml 
3. conda activate env_opf_dnn
4. juliaup add 1.6.7
5. juliaup default 1.6.7
6. julia --color=yes setup_julia_env.jl
7. cd to the parent folder (where we have src/ folder. You can do cd .. 3 times to do so.)
8. Now follow the running instructions to run the code. But before, read the notes below.

#### Note: 
In my Linux system, I need to use this command once in the beginning to get rid of the **C++ runtime mismatch** between **Julia 1.6’s old `libstdc++.so.6`** and the **newer conda libraries**
```
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1"
```


While running the code, if you get 'can't import torch (or something else)' error despite installing them, rebuilding pycall might help:

```julia
julia --project=src/baselines/OPF_DNN -e '
using Pkg;
ENV["PYTHON"] = joinpath(ENV["CONDA_PREFIX"], "bin", "python");
Pkg.build("PyCall");'
```


### Running Instructions
Generate training data:
```
python3 -m src.baselines.OPF_DNN.opf-datagen.opf_datagen --split "train" 
```

Generate Test data:
```
python3 -m src.baselines.OPF_DNN.opf-datagen.opf_datagen --split "test" --which_data "in_distribution"
```

Train and test model:
```
julia --project=src/baselines/OPF_DNN src/baselines/OPF_DNN/opf-dnn.jl --testdata "test_data_fuel_price_up.json"
```
Note: Change --testdata to be the testdata file name you want to test on. Find the files inside src/baselines/OPF_DNN/datasets/