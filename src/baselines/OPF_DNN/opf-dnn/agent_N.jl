using Printf, ProgressMeter, Statistics
using JSON
using PyCall

include("torch.jl")
include("networks.jl")
include("utils.jl")
include("datautils.jl")
include("constraints.jl")



""" OPF Agent class """
mutable struct OpfAgent
    config::Dict
    device::PyObject
    model::Any
    optimizer::Any
    loss::Dict

    train_loader::DataLoader
    test_loader::DataLoader
    constraints::Constraints
    datainfo::DataInfo
    verbose::Bool
    data_indexes::Any

    function OpfAgent(args)
        # Select Device
        device = torch.device(ifelse(!args["nocuda"] && torch.cuda.is_available(), "cuda", "cpu"))
        # Read Data and construct auxiliary structures
        data_train = read_data(args, "train")
        data_test = read_data(args, "test")
        #reindex, make it 1-based (function written in utils.jl)
        _reindex_branch_keys!(data_train)          # rename to L1.., T1..
        _ensure_complete_branch_dicts!(data_train) # fill any missing keys with defaults 
        
        _reindex_branch_keys!(data_test)          
        _ensure_complete_branch_dicts!(data_test)

        datainfo_train = DataInfo(data_train, args)
        constr   = Constraints(data_train, datainfo_train, args, device)

        # Load training Dataset
        (Sd_tr, Flows_tr, vm_tr, va_tr, pg_tr, qg_tr, data_indexes_tr) = load_naive_datasets(data_train, datainfo_train, args)
        train = DataLoader(Sd_tr, Flows_tr, vm_tr, va_tr, pg_tr, qg_tr,
                                 shuffle=true, batchsize=args["batchsize"])

        indims = Dict("Sd" => size(Sd_tr)[2])
        outdims = Dict("vm" => size(vm_tr)[2], "va" => size(va_tr)[2],
                       "pg" => size(pg_tr)[2], "qg" => size(qg_tr)[2])

        # Create Model and Set optimizers
        model = DeepFFNetPVNaive(indims, outdims, args["nettype"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args["lr"], betas=(0.9, 0.999))

        loss = Dict("vm" => F.mse_loss, "va" => F.mse_loss,
                    "pg" => F.mse_loss, "qg" => F.mse_loss,
                    "ohm" => F.mse_loss,
                    "vm-bnd" => partial(bound_penalty, constr.bnd["vm-bnd"]),
                    "va-bnd" => partial(bound_penalty, constr.bnd["va-bnd"]),
                    "pg-bnd" => partial(bound_penalty, constr.bnd["pg-bnd"]),
                    "flow-bnd" => partial(bound_penalty, constr.bnd["flow-bnd"]))

        #load test data 
        datainfo_test = DataInfo(data_test, args)
        (Sd, Flows, vm, va, pg, qg, data_indexes) = load_naive_datasets(data_test, datainfo_test, args)
        test = DataLoader(Sd, Flows, vm, va, pg, qg,
                                 shuffle=false, batchsize=args["batchsize"])
        #create a new agent with training datainfo                         
        new(args, device, model, optimizer, loss,
            train, test, constr, datainfo_train, args["verbose"],
            data_indexes_tr)
    end
end # module

# todo continue from here
function train(agent::OpfAgent)
    """
    Main training loop
    """
    nepochs = agent.config["nepochs"]
    device = agent.device
    batchsize = agent.config["batchsize"]

    _lkeys = ["vm", "va", "vm-bnd", "va-bnd", "pg", "pg-bnd",
              "ohm", "flow-bnd", "klc"]
    _losses = Dict(k => [] for k in _lkeys)
    L = Dict{String,Any}(k => nothing for k in _lkeys)
    con = agent.constraints

    #update every 10 seconds
    barlen=round(Int, nepochs*agent.train_loader.n / batchsize)
    p = Progress(barlen, dt=0.1, color=:blue)
    for epoch in 1:nepochs
        let Lepoch = Dict{String,Any}(k => [] for k in _lkeys)
            for (i, (sd, flow, vm, va, pg, qg)) in enumerate(agent.train_loader)
                (sd, flow, vm, va, pg, qg) = Ten(sd).to(device), Ten(flow).to(device),
                                         Ten(vm).to(device), Ten(va).to(device),
                                         Ten(pg).to(device), Ten(qg).to(device)

                """ Predict Voltage Magnitudes and Angle"""
                vm_pred, va_pred, pg_pred = agent.model(sd)

                """ Predict FLows from va and vm """
                flow_pred = get_Flows(con.olc, vm_pred, va_pred)
                oSf, oSt = get_complex_flows(flow_pred)

                """ Losses for vm """
                L["vm"] = agent.loss["vm"](vm_pred, vm)
                L["vm-bnd"]  = agent.loss["vm-bnd"](vm_pred)
                push!(_losses["vm"], L["vm"].item())
                push!(_losses["vm-bnd"], L["vm-bnd"].item())
                push!(Lepoch["vm-bnd"], L["vm-bnd"].item())

                """ Losses for va """
                L["va"] = agent.loss["va"](va_pred, va)
                L["va-bnd"]  = agent.loss["va-bnd"](va_pred)
                push!(_losses["va"], L["va"].item())
                push!(_losses["va-bnd"], L["va-bnd"].item())
                push!(Lepoch["va-bnd"], L["va-bnd"].item())

                """ Losses for pg """
                L["pg"] = agent.loss["pg"](pg_pred, pg)
                L["pg-bnd"]  = agent.loss["pg-bnd"](pg_pred)
                push!(_losses["pg"], L["pg"].item())
                push!(_losses["pg-bnd"], L["pg-bnd"].item())
                push!(Lepoch["pg-bnd"], L["pg-bnd"].item())

                """ Losses for Flows and RateA Losses """
                L["flow-bnd"] = agent.loss["flow-bnd"](oSf) + agent.loss["flow-bnd"](oSt)
                push!(_losses["flow-bnd"], L["flow-bnd"].item())
                push!(Lepoch["flow-bnd"], L["flow-bnd"].item())

                L["ohm"] = agent.loss["ohm"](flow_pred, flow)
                push!(_losses["ohm"], L["ohm"].item())
                push!(Lepoch["ohm"], L["ohm"].item())

                (pd, qd) = get_components_from_Sd(sd, agent.datainfo)
                (pf_pred, _, pt_pred, _) = get_components_from_Sij(flow_pred, agent.datainfo)
                L["klc"] = get_losses(agent.constraints.pbc, pd, pg_pred, vm_pred, pf_pred, pt_pred)
                push!(Lepoch["klc"], L["klc"].item())

                loss = get_losses(L, con, agent.config)
                backprop(agent, loss)

                !agent.verbose && next!(p)
                if agent.verbose && i % 100 == 0
                    println("Epoch: $epoch\t($i) ")
                    println("$(@sprintf("\tLosses: \tvm: %.6f \tbnd: %.6f", L["vm"].item(), L["vm-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tva: %.6f \tbnd: %.6f", L["va"].item(), L["va-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tpg: %.6f \tbnd: %.6f", L["pg"].item(), L["pg-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tflow: %.6f \tbnd: %.6f", L["ohm"].item(), L["flow-bnd"].item()))")
                    println("$(@sprintf("\tLosses: \tPBC: %.6f", L["klc"].item()))")
                end
            end

            # Lagrangian Step
            update_allλ!(con, Lepoch)
            GC.gc(false)
        end
    end
    return _losses
end

function get_losses(L, con, config)
    # version = "[p]g [v]m [v]a f[low] o[hm] k[lc]"
    _losses = L["vm"] + L["va"] + L["pg"]
    if config["use-constraints"]
        if config["use-dual-update"]
            for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]
                _losses += con.λ[ctype] * L[ctype]
            end
        else
            for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]
                _losses += L[ctype]
            end
        end
    end
    return _losses
end


######################## AFIA 
function test_pf(agent::OpfAgent; dataset_json_path::AbstractString,
                 py_src_root::AbstractString=".", verbose::Bool=false)

    # --- helpers ---------------------------------------------------------
    toA(x, device) = toarray(Ten(x).to(device))

    function _infer(agent, sd_t)
        if hasproperty(agent, :model)
            return agent.model(sd_t)
        elseif hasproperty(agent, :policy)
            return agent.policy(sd_t)
        else
            return agent(sd_t)
        end
    end

    function _unpack(out)
        if out isa Dict
            vm = haskey(out, "vm") ? out["vm"] : out[:vm]
            va = haskey(out, "va") ? out["va"] : out[:va]
            pg = haskey(out, "pg") ? out["pg"] : out[:pg]
            return vm, va, pg
        elseif out isa NamedTuple
            vm = haskey(out, :vm) ? out.vm : out[:vm]
            va = haskey(out, :va) ? out.va : out[:va]
            pg = haskey(out, :pg) ? out.pg : out[:pg]
            return vm, va, pg
        elseif out isa Tuple
            @assert length(out) ≥ 3 "Expected at least 3 outputs (vm, va, pg)"
            return out[1], out[2], out[3]
        else
            error("Unrecognized model output type: $(typeof(out))")
        end
    end

    # --- Load dataset JSON ONCE (for meta/net_paths + consistent OLC for flows) ---
    ds = JSON.parsefile(String(dataset_json_path))
    meta= ds["meta"]
    net_paths = haskey(ds, "net_paths") ? ds["net_paths"] : meta["net_paths"]

    data_eval = ds
    _reindex_branch_keys!(data_eval)
    _ensure_complete_branch_dicts!(data_eval)

    datainfo_eval = DataInfo(data_eval, agent.config)
    constr_eval = Constraints(data_eval, datainfo_eval, agent.config, agent.device)

    # Use this OLC for computing flows to match this dataset's flow_gt
    olc_for_flow = constr_eval.olc

    # --- Relative L1 (%) accumulators ---------------------------------------
    eps = 1e-12
    sae= Dict("flow"=>0.0, "vm"=>0.0, "va"=>0.0, "pg"=>0.0)   # sum |pred-gt|
    sabs= Dict("flow"=>0.0, "vm"=>0.0, "va"=>0.0, "pg"=>0.0)   # sum |gt|

    # --- Build Python loader ONCE  ----------------------------
    pylist = pybuiltin("list")
    loader = pylist()  # list of (xb, yb); yb unused by evaluator when collect_norm_mse=false

    # --- evaluation loop -----------------------------------------------------
    for (sd, flow_gt, vm_gt, va_gt, pg_gt, qg_gt) in agent.test_loader
        # keep a CPU tensor for the python evaluator loader
        sd_cpu = Ten(sd).detach()
        push!(loader, (sd_cpu, sd_cpu))

        # move to device for inference
        sd_t = Ten(sd).to(agent.device)

        vm_gtA = toA(vm_gt, agent.device)
        va_gtA = toA(va_gt, agent.device)
        pg_gtA = toA(pg_gt, agent.device)
        flow_gtA = toA(flow_gt, agent.device)

        out = _infer(agent, sd_t)
        vm_pred_t, va_pred_t, pg_pred_t = _unpack(out)

        vm_predA = toA(vm_pred_t, agent.device)
        va_predA = toA(va_pred_t, agent.device)
        pg_predA = toA(pg_pred_t, agent.device)

        # compute flows ONCE (no redundancy)
        flow_pred_t = get_Flows(olc_for_flow, vm_pred_t, va_pred_t)
        flow_predA= toA(flow_pred_t, agent.device)

        # accumulate
        sae["vm"] += sum(abs.(vm_predA .- vm_gtA));     sabs["vm"]   += sum(abs.(vm_gtA))
        sae["va"] += sum(abs.(va_predA .- va_gtA));     sabs["va"]   += sum(abs.(va_gtA))
        sae["pg"] += sum(abs.(pg_predA .- pg_gtA));     sabs["pg"]   += sum(abs.(pg_gtA))
        sae["flow"]+= sum(abs.(flow_predA .- flow_gtA)); sabs["flow"] += sum(abs.(flow_gtA))
    end

    # --- Make src.* importable ---
    # sys = pyimport("sys")
    # if !(abspath(py_src_root) in collect(sys."path"))
    #     push!(sys."path", abspath(py_src_root))
    # end

    # --- Python deps ---
    torch = pyimport("torch")
    codes_root = get(ENV, "CODES_ACOPF_ROOT", "")
    isempty(codes_root) && error("CODES_ACOPF_ROOT not set")

    # sanity check on Julia side
    @assert isdir(joinpath(codes_root, "src")) "No src/ folder under CODES_ACOPF_ROOT"
    @assert isfile(joinpath(codes_root, "src", "global_evaluation.py")) "global_evaluation.py not found under CODES_ACOPF_ROOT/src"

    sys = pyimport("sys")
    pushfirst!(sys."path", codes_root)   # make sure it's first
    
    ge = pyimport("src.global_evaluation")
    println("PY ge.__file__ = ", ge."__file__")
    gvars = pyimport("src.global_vars")

    # --- Python helpers for PFEvaluator ---
    py"""
    import numpy as np, torch
    import src.global_evaluation as ge

    def _denorm_fn(y_norm, meta, device):
        return y_norm

    def _apply_pred_fn(base_net, meta, x_i, y_pred_i, denorm_fn, device):
        if isinstance(y_pred_i, (tuple, list)):
            vm_t, va_t, pg_t = y_pred_i
        else:
            raise TypeError("Expected y_pred_i as (vm, va, pg)")

        vm = vm_t.detach().cpu().numpy().astype(float).ravel()
        va = va_t.detach().cpu().numpy().astype(float).ravel()
        pg = pg_t.detach().cpu().numpy().astype(float).ravel()

        bus_order = list(meta["bus_index_order"])
        gen_order = list(meta["gen_index_order"])

        if hasattr(base_net, "ext_grid") and len(base_net.ext_grid):
            for idx, row in base_net.ext_grid.iterrows():
                b = int(row["bus"])
                if b in bus_order:
                    j = bus_order.index(b)
                    base_net.ext_grid.at[idx, "vm_pu"] = float(vm[j])
                    if "va_degree" in base_net.ext_grid.columns:
                        base_net.ext_grid.at[idx, "va_degree"] = float(np.degrees(va[j]))

        if hasattr(base_net, "gen") and len(base_net.gen):
            for gi, row in base_net.gen.iterrows():
                b = int(row["bus"])
                if b in bus_order:
                    j = bus_order.index(b)
                    base_net.gen.at[gi, "vm_pu"] = float(vm[j])

        if hasattr(base_net, "gen") and len(base_net.gen):
            for k, gi in enumerate(gen_order):
                if gi in base_net.gen.index:
                    pmin = float(base_net.gen.at[gi, "min_p_mw"])
                    pmax = float(base_net.gen.at[gi, "max_p_mw"])
                    pval = float(np.clip(pg[k], pmin, pmax))
                    base_net.gen.at[gi, "p_mw"] = pval
                else:
                    gi2 = int(gi)
                    if 0 <= gi2 < len(base_net.gen):
                        pmin = float(base_net.gen.iloc[gi2]["min_p_mw"])
                        pmax = float(base_net.gen.iloc[gi2]["max_p_mw"])
                        pval = float(np.clip(pg[k], pmin, pmax))
                        base_net.gen.iloc[gi2, base_net.gen.columns.get_loc("p_mw")] = pval

        return base_net

    def _safe_is_torch(self, obj):
        try:
            mod = getattr(getattr(obj, '__class__', object), '__module__', '')
        except Exception:
            return False
        return (ge.torch is not None) and ('torch' in str(mod))

    ge.PFEvaluator._is_torch = _safe_is_torch

    def _infer_batched(model, xb, device):
        if hasattr(xb, "to"):
            xb = xb.to(device)
        y = model(xb)
        if not (isinstance(y, (tuple, list)) and len(y) == 3):
            raise TypeError("model(x) must return (vm, va, pg)")
        vm, va, pg = y
        B = int(vm.shape[0])
        return [(vm[i], va[i], pg[i]) for i in range(B)]
    """

    runpp_kwargs = Dict("enforce_q_lims" => true, "calculate_voltage_angles" => true)
    evaluator = ge.PFEvaluator(
        py"_denorm_fn",
        py"_apply_pred_fn",
        py"_infer_batched",
        gvars.feas_tol, gvars.max_pen, gvars.ctol,
        runpp_kwargs
    )

    println("Built Python loader with ", Int(py"len"(loader)), " items.")

    pyres = evaluator.evaluate(agent.model, loader, meta, net_paths, agent.device,
                               false, verbose)

    costs = Float64.(pyres["costs"])
    feas = Bool.(pyres["hard_feasible_flags"])
    feas_rate = Float64(pyres["feasibility_rate_%"])
    
    avg_cost = isempty(costs) ? NaN : mean(costs)

    infer_time_per_sample = haskey(pyres, "infer_time_per_sample_s") ? Float64(pyres["infer_time_per_sample_s"]) : NaN
    infer_kept_count      = haskey(pyres, "infer_time_kept_n") ? Int(pyres["infer_time_kept_n"]) : 0

    return Dict(
        "n_samples" => Int(pyres["n_samples"]),
        "costs" => costs,
        "avg_cost" => avg_cost,                       # over all samples (feasible or not)
        "hard_feasible_flags" => feas,
        "feasibility_rate_%" => feas_rate,
        "infer_time_per_sample_s" => infer_time_per_sample,
        "infer_time_kept_n" => infer_kept_count,
    )
end