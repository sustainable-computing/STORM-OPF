"""
	OPF-DNN
	Author: Ferdinando Fioretto
	Date:   Sept. 1, 2019

	Reference: Ferdinando Fioretto, Terrence W. K. Mak, Pascal Van Hentenryck:
		 Predicting AC Optimal Power Flows: Combining Deep Learning and
		 Lagrangian Dual Methods. CoRR abs/1909.10461 (2019)
	Modified by:  Afia Afrin
	Date:  Nov 27, 2025
"""

using PyCall
include("opf-dnn/utils.jl")
include("opf-dnn/datautils.jl")
torch = pyimport("torch")
write_loadflow_results = true
write_test_results     = false
write_training_losses  = false
write_summary_results  = false

# Parse Arguments
args = parse_commandline()
fix_random_params(args["seed"])

if args["use-state"]		#use hot-start case
	include("opf-dnn/agent_S.jl")	
else  #default case, don't use hot-start case
	include("opf-dnn/agent_N.jl")
end

# Create agent and Train it
println("creating agent")
agent = OpfAgent(args)

if args["retrain"]
    println("training started")
    losses = train(agent)
    save_model(agent, args)
    println("training ended")
else
    load_model!(agent, args)  #load pre-saved trained model
end


#########
# Get and save Results  
########
   
test_data_path = joinpath("src", "baselines", "OPF_DNN", "datasets", args["netname"], args["testdata"])

results = test_pf(agent;
    dataset_json_path = test_data_path,
    py_src_root = "src",   # so PyCall can import src.global_evaluation & src.util
    verbose = true
)
println("ABOUT TO SAVE: keys=", collect(keys(results)))

# Convert Julia Dict -> Python dict
py_results = PyDict(results)  # from PyCall

# Save as pickle
results_dir= joinpath("results", args["netname"] )
which_data = split(args["testdata"], "test_data_")[2]
which_data= split(which_data, ".")[1]
fname = joinpath(results_dir, "OPFDNN_result_$(which_data).pkl")
const py_pickle = pyimport("pickle")
open(fname, "w") do io
    py_pickle["dump"](py_results, io)
end

#@show results["infer_time_per_sample_s"]