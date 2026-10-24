using Printf, ArgParse, Statistics, ProgressMeter
include("torch.jl")

# utils.jl

# Ensure L/T keys are 1-based & contiguous everywhere (constraints, experiments, AND network)
function _reindex_branch_keys!(data::Dict{String,Any})
    # --- 1) Reindex the network branch table to L#/T# ---
    if haskey(data, "branch") && data["branch"] isa Dict
        br = data["branch"]::Dict{String,Any}

        # classify transformer vs line (fallback: tap/shift != default => transformer)
        is_tr(row) = get(row, "transformer", get(row, "tap", 1.0) != 1.0 || get(row, "shift", 0.0) != 0.0)

        old_keys_sorted = sort(collect(keys(br)))  # stable order
        L_old, T_old = String[], String[]
        for k in old_keys_sorted
            (is_tr(br[k]) ? push!(T_old, k) : push!(L_old, k))
        end

        # mapping for branch table
        branch_map = Dict{String,String}()
        for (i,k) in enumerate(L_old); branch_map[k] = "L$(i)"; end
        for (i,k) in enumerate(T_old); branch_map[k] = "T$(i)"; end

        # apply to data["branch"]
        new_br = Dict{String,Any}()
        for (ok, row) in br
            new_br[branch_map[ok]] = row
        end
        data["branch"] = new_br

        # We'll also use branch_map to normalize constraints/experiments below
    else
        branch_map = Dict{String,String}()  # nothing to do
    end

    # --- 2) Make constraints keys contiguous L#/T# (respecting branch_map when possible) ---
    cons = haskey(data, "constraints") ? (data["constraints"]::Dict{String,Any}) : Dict{String,Any}()

    # helper: rebuild dict with mapping; if key not found in mapping, keep as-is
    remap!(d::Dict{String,Any}, map::Dict{String,String}) = begin
        newd = Dict{String,Any}()
        for (k,v) in d
            nk = get(map, k, k)  # try branch_map first
            newd[nk] = v
        end
        empty!(d); merge!(d, newd); d
    end

    if haskey(cons, "rate_a") && cons["rate_a"] isa Dict
        remap!(cons["rate_a"], branch_map)
    end
    if haskey(cons, "line_rx") && cons["line_rx"] isa Dict
        remap!(cons["line_rx"], branch_map)
    end
    if haskey(cons, "line_bg") && cons["line_bg"] isa Dict
        remap!(cons["line_bg"], branch_map)
    end

    # --- 3) Remap experiments flow keys to match (pf/pt/qf/qt) ---
    if haskey(data, "experiments") && data["experiments"] isa Vector
        for exp in data["experiments"]
            for fld in ("pf","pt","qf","qt")
                if haskey(exp, fld) && exp[fld] isa Dict
                    remap!(exp[fld], branch_map)
                end
            end
        end
    end

    return data
end


# Fill missing keys across all branch-related dicts
function _ensure_complete_branch_dicts!(data::Dict{String,Any})
    cons = data["constraints"]::Dict{String,Any}
    # expected branch set from rate_a
    branches = collect(keys(cons["rate_a"]::Dict{String,Any}))

    # make sure dicts exist
    for fld in ("line_rx","line_bg")
        get!(cons, fld, Dict{String,Any}())
    end

    # defaults
    for k in branches
        # if a key is missing, supply a safe default
        cons["rate_a"][k]  = get(cons["rate_a"],  k, 1.0e6)
        cons["line_rx"][k] = get(cons["line_rx"], k, (0.0, 0.0))
        cons["line_bg"][k] = get(cons["line_bg"], k, (0.0, 0.0))
    end

    # per-experiment flows (ohms law code may or may not use them, but be safe)
    for exp in data["experiments"]
        for fld in ("pf","pt","qf","qt")
            d = get!(exp, fld, Dict{String,Any}())
            for k in branches
                d[k] = get(d, k, 0.0)
            end
        end
    end
    return data
end


""" Parse Arguments """
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netpath" # do not change
            help = "The path to the input networks (.m)"
            arg_type = String
            default = "src/scenarios/"
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "case14"
        "--traindata" # can read pickle files
            help = "The name of the input file, within the netname folder"
            default = "training_data_in_distribution.json"
        "--testdata" # can read pickle files
            help = "The name of the test data file, within the netname folder"
            default = "test_data_in_distribution.json"
        "--out-suffix", "-s" ## also used for version
            help = "The suffix given to the output file to identify a given program variant"
            default = nothing
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
            default = 50
        "--batchsize", "-b"
            help = "The size of the batch"
            arg_type = Int
            default = 1500
        "--state-distance"
            help = "The distance, in percentage, between each two network
                    states, used in the construction of hot-start states
                    training data"
            arg_type = Float64
            default = 1.0 # = 1 %
        "--seed"
            arg_type = Int
            default = 1234
        # "--split"
        #     help = "Train split in (0, 1). The rest is given to Test"
        #     arg_type = Float64
        #     default = 0.8
        "--lr"
            help = "The learning rate"
            default = 0.001
        "--dur"
            help = "Dual update rate"
            default = 0.01
        "--traindata-size"
            help = "maxmium traindata size"
            default = 100000
    end
    return parse_args(s)
end

""" Sum a vector of numbers if the vector is non empty, otherwise returns 0 """
function sum0(vec)
    return length(vec) > 0 ? sum(vec) : 0.0
end

""" Backprop function """
function backprop(agent, loss, retain_graph=false)
    agent.optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    agent.optimizer.step()
end

"""" Return pd[t-1], pd[t], qd[t-1], qd[t] for setpoint test """
function get_combo_Sd(x, nloads, numpy)
    if numpy
        return x.narrow(1, 0, nloads).cpu().numpy(),          # pd[t-1]
               x.narrow(1, nloads, nloads).cpu().numpy(),     # pd[t]
               x.narrow(1, 2*nloads, nloads).cpu().numpy(),   # qd[t-1]
               x.narrow(1, 3*nloads, nloads).cpu().numpy()    # qd[t]
    else
        return x.narrow(1, 0, nloads),          # pd[t-1]
               x.narrow(1, nloads, nloads),     # pd[t]
               x.narrow(1, 2*nloads, nloads),   # qd[t-1]
               x.narrow(1, 3*nloads, nloads)    # qd[t]
    end
end

""" Given a file name (may include its path), returns its extension """
extension(url::String) = try match(r"\.[A-Za-z0-9]+$", url).match catch nothing end

""" Given a file name (may include its path), returns its name """
filename(url::String) = try match(r"([A-Za-z0-9]+-*+_*)*", url).match catch nothing end

function get_file_name(config, type, prefix=nothing)
    @assert type in ["results", "model", "plot"]

    _PATH = "src/baselines/OPF_DNN/model/" * config["netname"] * "/"
    _nettype = args["nettype"] * "-"
    _suffix = config["out-suffix"] == nothing ? "" : "-" * config["out-suffix"]
    _prefix = prefix == nothing ? "" : "-" * prefix

    if type == "results"
        _fname, _ext = filename(config["results-outfile"]), extension(config["results-outfile"])
    elseif type == "plot"
        _fname, _ext = filename(config["plot-outfile"]), extension(config["plot-outfile"])
    elseif type == "model"
        _fname, _ext = "model", ".mdl"
    end

    return _PATH * _nettype * _fname * _prefix * _suffix * _ext
end
