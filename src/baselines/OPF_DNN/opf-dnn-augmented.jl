using ArgParse
using Random
using JSON
using Statistics
using PyCall

const py_pickle = pyimport("pickle")

function parse_commandline_augmented()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netpath"
            help = "The path to the input networks (.m)"
            arg_type = String
            default = "src/scenarios/"
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "case14"
        "--traindata"
            help = "The nominal training data file, within the netname folder"
            default = "training_data_in_distribution.json"
        "--stress-traindata"
            help = "The stressed training data file, within the netname folder"
            default = "training_data_stressed.json"
        "--testdata"
            help = "The name of the test data file, within the netname folder"
            default = "test_data_in_distribution.json"
        "--test-all"
            help = "Run on all pre-generated test_data_*.json files in the dataset folder"
            default = false
            action = :store_true
        "--retrain"
            help = "Train again the augmented model, initialized from baseline"
            default = false
            action = :store_true
        "--out-suffix", "-s"
            help = "Suffix given to saved model/results for the augmented variant"
            default = "augmented"
        "--plot-outfile", "-p"
            help = "The name of the result file, within the netname folder"
            default = "losses.png"
        "--results-outfile", "-r"
            help = "The name of the result file, within the netname folder"
            default = "results.pkl"
        "--use-state"
            help = "Exploit hot-start state"
            default = false
            action = :store_true
        "--use-constraints"
            help = "Use Lagrangian constraint penalties"
            default = true
            action = :store_true
        "--use-dual-update"
            help = "Use Lagrangian dual update"
            default = false
            action = :store_true
        "--nocuda"
            help = "Do not use CUDA"
            action = :store_true
        "--verbose"
            action = :store_true
        "--nettype", "-t"
            help = "enc[oder] or dec[oder]"
            arg_type = String
            default = "dec"
        "--nepochs", "-e"
            help = "The number of epochs"
            arg_type = Int
            default = 10
        "--batchsize", "-b"
            help = "The size of the batch"
            arg_type = Int
            default = 3072
        "--warmup-epochs"
            help = "Epochs with nominal-only training"
            arg_type = Int
            default = 3
        "--ramp-epochs"
            help = "Epochs used to ramp stressed samples into training"
            arg_type = Int
            default = 5
        "--state-distance"
            help = "The distance, in percentage, between each two network states"
            arg_type = Float64
            default = 1.0
        "--seed"
            arg_type = Int
            default = 1234
        "--lr"
            help = "The learning rate"
            default = 0.0001
        "--dur"
            help = "Dual update rate"
            default = 0.01
        "--traindata-size"
            help = "maximum traindata size"
            default = 100000
        "--py-src-root"
            help = "Python src root for test_pf global evaluation helpers"
            arg_type = String
            default = "."
    end
    return parse_args(s)
end

global args = parse_commandline_augmented()

if args["use-state"]
    include(joinpath("opf-dnn", "agent_S.jl"))
else
    include(joinpath("opf-dnn", "agent_N.jl"))
end

function dataset_dir(config)
    joinpath("src", "baselines", "OPF_DNN", "datasets", config["netname"])
end

function load_json_dict(path::AbstractString)
    data = JSON.parsefile(String(path))
    _reindex_branch_keys!(data)
    _ensure_complete_branch_dicts!(data)
    return data
end

function load_model_from_file!(agent::OpfAgent, model_file::AbstractString; set_eval::Bool=false)
    println("[init] Loading model: $(model_file)")
    state_dict = torch.load(model_file, map_location=agent.device)
    agent.model.load_state_dict(state_dict)
    if set_eval
        agent.model.eval()
    else
        agent.model.train()
    end
    return nothing
end

function maybe_load_saved_augmented_model!(agent::OpfAgent, config::Dict{String,Any})
    aug_file = get_file_name(config, "model")
    if !config["retrain"] && isfile(aug_file)
        println("[init] Found saved augmented model, skipping training: $(aug_file)")
        load_model_from_file!(agent, aug_file; set_eval=true)
        return true
    end
    return false
end

function load_training_tuple(data::Dict{String,Any}, config::Dict{String,Any})
    datainfo = DataInfo(data, config)
    if config["use-state"]
        tup = load_setpoint_datasets_Perc(data, datainfo, config)
        return (tup[1], tup[2], tup[3], tup[4], tup[5], tup[6], tup[7], tup[8])
    else
        tup = load_naive_datasets(data, datainfo, config)
        return (tup[1], tup[2], tup[3], tup[4], tup[5], tup[6])
    end
end

function make_loader_from_tuple(data_tuple::Tuple, batchsize::Int; shuffle::Bool=true)
    DataLoader(data_tuple...; batchsize=batchsize, shuffle=shuffle)
end

function epoch_stress_count(epoch::Int, warmup_epochs::Int, ramp_epochs::Int, n_stress_total::Int)
    if epoch <= warmup_epochs
        return 0
    end
    if ramp_epochs <= 0
        return n_stress_total
    end
    prog = min(1.0, float(epoch - warmup_epochs) / float(ramp_epochs))
    return Int(round(prog * n_stress_total))
end

function sample_epoch_dataset_full_union(nominal::Tuple, stressed::Tuple, n_stress_use::Int, seed::Int)
    n_nom = size(nominal[1], 1)
    n_str = size(stressed[1], 1)
    n_stress_use = clamp(n_stress_use, 0, n_str)

    if n_stress_use == 0
        return nominal, n_nom, 0
    end

    rng = MersenneTwister(seed)
    str_idx = randperm(rng, n_str)[1:n_stress_use]
    merged = ntuple(i -> vcat(nominal[i], stressed[i][str_idx, :]), length(nominal))
    perm = randperm(rng, size(merged[1], 1))
    mixed = ntuple(i -> merged[i][perm, :], length(merged))
    return mixed, n_nom, n_stress_use
end

function set_test_loader!(agent::OpfAgent, config::Dict{String,Any}, testfile::AbstractString)
    config["testdata"] = String(testfile)
    data_test = read_data(config, "test")
    _reindex_branch_keys!(data_test)
    _ensure_complete_branch_dicts!(data_test)
    datainfo_test = DataInfo(data_test, config)
    tup = load_naive_datasets(data_test, datainfo_test, config)
    agent.test_loader = DataLoader(
        tup[1], tup[2], tup[3], tup[4], tup[5], tup[6];
        shuffle=false, batchsize=config["batchsize"]
    )
    return nothing
end

function train_augmented!(agent::OpfAgent, stressed_tuple::Tuple, config::Dict{String,Any})
    nominal_tuple = agent.train_loader.dataset
    total_epochs = config["nepochs"]
    warmup_epochs = config["warmup-epochs"]
    ramp_epochs = config["ramp-epochs"]
    batchsize = config["batchsize"]
    base_seed = config["seed"]
    n_stress_total = size(stressed_tuple[1], 1)

    original_nepochs = agent.config["nepochs"]
    agent.config["nepochs"] = 1

    history = Dict(
        "epoch" => Int[],
        "n_nominal" => Int[],
        "n_stress" => Int[],
        "stress_fraction_of_epoch" => Float64[]
    )

    for epoch in 1:total_epochs
        n_stress_use = epoch_stress_count(epoch, warmup_epochs, ramp_epochs, n_stress_total)
        epoch_tuple, n_nominal, n_stress = sample_epoch_dataset_full_union(
            nominal_tuple, stressed_tuple, n_stress_use, base_seed + epoch
        )
        n_total = size(epoch_tuple[1], 1)
        frac = n_total == 0 ? 0.0 : (n_stress / n_total)

        println(
            "[Augmented OPF-DNN] epoch $(epoch)/$(total_epochs): " *
            "nominal=$(n_nominal), stress=$(n_stress), total=$(n_total), " *
            "stress_frac=$(round(frac, digits=4))"
        )

        agent.train_loader = make_loader_from_tuple(epoch_tuple, batchsize; shuffle=true)
        train(agent)

        push!(history["epoch"], epoch)
        push!(history["n_nominal"], n_nominal)
        push!(history["n_stress"], n_stress)
        push!(history["stress_fraction_of_epoch"], frac)
    end

    agent.config["nepochs"] = original_nepochs
    agent.train_loader = make_loader_from_tuple(nominal_tuple, batchsize; shuffle=true)
    return history
end

function save_test_result(result::Dict{String,Any}, config::Dict{String,Any}, testfile::AbstractString)
    results_dir = joinpath("results", config["netname"])
    mkpath(results_dir)

    which_data = split(String(testfile), "test_data_")[2]
    which_data = split(which_data, ".")[1]

    fname = joinpath(results_dir, "OPFDNN_augmented_result_$(which_data).pkl")

    open(fname, "w") do io
        py_pickle["dump"](result, io)
    end

    println("Writing: $(fname)")
end

function test_files_for_run(config)
    ddir = dataset_dir(config)
    if config["test-all"]
        files = sort(filter(f -> startswith(f, "test_data_") && endswith(f, ".json"), readdir(ddir)))
        isempty(files) && error("No test_data_*.json files found in $(ddir)")
        return files
    else
        return [config["testdata"]]
    end
end

function baseline_model_file(config::Dict{String,Any}, prefix=nothing)
    aug_file = get_file_name(config, "model", prefix)
    stem, ext = splitext(aug_file)
    suffix = "-" * String(config["out-suffix"])

    if endswith(stem, suffix)
        return stem[1:end-length(suffix)] * ext
    end

    return replace(aug_file, r"-augmented(?=\.[^.]+$)" => "")
end

function load_initial_model_for_augmented!(agent::OpfAgent, config::Dict{String,Any}, prefix=nothing)
    init_file = baseline_model_file(config, prefix)
    isfile(init_file) || error("Missing initialization model file: $(init_file)")

    if config["retrain"]
        println("[init] Re-training augmented OPF-DNN from BASELINE model: $(init_file)")
    else
        println("[init] Training augmented OPF-DNN from BASELINE model: $(init_file)")
    end

    load_model_from_file!(agent, init_file; set_eval=false)
    return nothing
end

function main()
    Random.seed!(args["seed"])

    agent = OpfAgent(args)

    trained_already = maybe_load_saved_augmented_model!(agent, args)

    if !trained_already
        load_initial_model_for_augmented!(agent, args)

        stress_json = joinpath(dataset_dir(args), args["stress-traindata"])
        isfile(stress_json) || error("Missing stressed training JSON: $(stress_json)")

        stress_data = load_json_dict(stress_json)
        stressed_tuple = load_training_tuple(stress_data, args)

        _history = train_augmented!(agent, stressed_tuple, args)

        save_model(agent, args)
        agent.model.eval()
    end

    for (i, testfile) in enumerate(test_files_for_run(args))
        if !(i == 1 && String(testfile) == String(args["testdata"]))
            set_test_loader!(agent, args, testfile)
        end

        dataset_json_path = joinpath(dataset_dir(args), testfile)
        result = test_pf(
            agent;
            dataset_json_path=dataset_json_path,
            py_src_root=args["py-src-root"],
            verbose=args["verbose"]
        )
        save_test_result(result, args, testfile)
    end
end

main()