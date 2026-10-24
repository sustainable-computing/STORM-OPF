using Printf, ProgressMeter, Statistics
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

        datainfo = DataInfo(data_train, args)
        constr   = Constraints(data_train, datainfo, args, device)
        # Load Datasets
        # (Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out, data_indexes) = load_setpoint_datasets(data, datainfo, args)
        (Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out, data_indexes) = load_setpoint_datasets_Perc(data_train, datainfo, args)

        train = DataLoader(Sd, vm_in, va_in, pg_in, Flows_out, vm_out, va_out, pg_out,
                                 shuffle=false, batchsize=args["batchsize"])

        indims = Dict("Sd" => size(Sd)[2], "vm" => size(vm_in)[2],
                      "va" => size(va_in)[2], "pg" => size(pg_in)[2])
        outdims = Dict("vm" => size(vm_out)[2], "va" => size(va_out)[2],
                       "pg" => size(pg_out)[2])

        # Create Model and Set optimizers
        println("l51: creating model")
        model = DeepFFNetPV(indims, outdims, args["nettype"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args["lr"], betas=(0.9, 0.999))

        loss = Dict("vm" => F.mse_loss, "va" => F.mse_loss,
                    "pg" => F.mse_loss, "ohm" => F.mse_loss,
                    "vm-bnd" => partial(bound_penalty, constr.bnd["vm-bnd"]),
                    "va-bnd" => partial(bound_penalty, constr.bnd["va-bnd"]),
                    "pg-bnd" => partial(bound_penalty, constr.bnd["pg-bnd"]),
                    "flow-bnd" => partial(bound_penalty, constr.bnd["flow-bnd"])
                    )
        datainfo_test = DataInfo(data_test, args)
        (Sd, Flows, vm, va, pg, qg, data_indexes_test) = load_naive_datasets(data_test, datainfo_test, args)
        test = DataLoader(Sd, Flows, vm, va, pg, qg,
                                 shuffle=false, batchsize=args["batchsize"])
        println('l66: will create new agent instance')
        new(args, device, model, optimizer, loss,
            train, test, constr, datainfo, args["verbose"],
            data_indexes)
        println("l70: opfAgent done")
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
            for (i, (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg)) in enumerate(agent.train_loader)
                (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg) = Ten(sd).to(device),
                    Ten(ivm).to(device), Ten(iva).to(device), Ten(ipg).to(device),
                    Ten(oflow).to(device), Ten(Δ_vm).to(device), Ten(Δ_va).to(device),
                    Ten(Δ_pg).to(device)

                """ Predict Voltage Magnitudes and Angle"""
                Δ_vm_pred, Δ_va_pred, Δ_pg_pred = agent.model(sd, ivm, iva, ipg)
                vm_pred, va_pred, pg_pred = ivm + Δ_vm_pred, iva + Δ_va_pred, ipg + Δ_pg_pred
                #vm_pred, va_pred, pg_pred = vm_pred.to(device), va_pred.to(device), pg_pred.to(device)

                """ Predict FLows from va and vm """
                oflow_pred = get_Flows(agent.constraints.olc, vm_pred, va_pred)
                oSf, oSt = get_complex_flows(oflow_pred)

                """ Losses for vm """
                L["vm"] = agent.loss["vm"](Δ_vm_pred, Δ_vm)
                L["vm-bnd"]  = agent.loss["vm-bnd"](vm_pred)
                push!(_losses["vm"], L["vm"].item())
                push!(_losses["vm-bnd"], L["vm-bnd"].item())
                push!(Lepoch["vm-bnd"], L["vm-bnd"].item())

                """ Losses for va """
                L["va"] = agent.loss["va"](Δ_va_pred, Δ_va)
                L["va-bnd"]  = agent.loss["va-bnd"](va_pred)
                push!(_losses["va"], L["va"].item())
                push!(_losses["va-bnd"], L["va-bnd"].item())
                push!(Lepoch["va-bnd"], L["va-bnd"].item())

                """ Losses for pg """
                L["pg"] = agent.loss["pg"](Δ_pg_pred, Δ_pg)
                L["pg-bnd"]  = agent.loss["pg-bnd"](pg_pred)
                push!(_losses["pg"], L["pg"].item())
                push!(_losses["pg-bnd"], L["pg-bnd"].item())
                push!(Lepoch["pg-bnd"], L["pg-bnd"].item())

                """ Losses for Flows and RateA Losses """
                L["flow-bnd"] = agent.loss["flow-bnd"](oSf) + agent.loss["flow-bnd"](oSt)
                push!(_losses["flow-bnd"], L["flow-bnd"].item())
                push!(Lepoch["flow-bnd"], L["flow-bnd"].item())

                L["ohm"] = agent.loss["ohm"](oflow_pred, oflow)
                push!(_losses["ohm"], L["ohm"].item())
                push!(Lepoch["ohm"], L["ohm"].item())

                """ Power blanace constraint losses """
                (_, pd_o, _, _) = get_components_from_Sd(sd, agent.datainfo)
                (pf_pred, _, pt_pred, _) = get_components_from_Sij(oflow_pred, agent.datainfo)
                L["klc"] = get_losses(agent.constraints.pbc, pd_o, pg_pred, vm_pred, pf_pred, pt_pred)
                push!(_losses["klc"], L["klc"].item())
                push!(Lepoch["klc"], L["klc"].item())

                loss = get_losses(L, con, agent.config)
                backprop(agent, loss)

                !agent.verbose && next!(p)
                #if agent.verbose && i % 100 == 0
                if i%100 == 0
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
            @show [con.λ[ctype] for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]]
            GC.gc(false)
        end
    end
    return _losses
end

function get_losses(L, con, config)
    _losses = L["vm"] + L["va"] + L["pg"]
    if config["use-constraints"]
        if config["use-dual-update"]
            for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]
                _losses += L[ctype]
            end
        else
            for ctype in ["pg-bnd", "vm-bnd", "va-bnd", "flow-bnd", "ohm", "klc"]
                _losses += con.λ[ctype] * L[ctype]
            end
        end
    end
    return _losses
end

# function test(agent::OpfAgent)
#     results = Dict()
#     N = agent.test_loader.n
#     err = Dict("vm" => [], "va" => [], "pg" => [],
#                "ohm" => [],  # error-pred
#                "klc" => [])

#    batchsize = agent.config["batchsize"]
#    barlen=round(Int, agent.test_loader.n / batchsize)
#    p = Progress(barlen, dt=1, color=:blue)
#     let device = agent.device
#         @pywith torch.no_grad() begin
#         for (i, (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg)) in enumerate(agent.test_loader)
#             # Update iterators in auxilary structures
#             #update_iterators(agent, "train")
#             (sd, ivm, iva, ipg, oflow, Δ_vm, Δ_va, Δ_pg) = Ten(sd).to(device),
#                 Ten(ivm).to(device), Ten(iva).to(device), Ten(ipg).to(device),
#                 Ten(oflow).to(device), Ten(Δ_vm).to(device), Ten(Δ_va).to(device),
#                 Ten(Δ_pg).to(device)

#                 Δ_vm_pred, Δ_va_pred, Δ_pg_pred = agent.model(sd, ivm, iva, ipg)
#                 vm_pred, va_pred, pg_pred = ivm + Δ_vm_pred, iva + Δ_va_pred, ipg + Δ_pg_pred

#                 vm_pred = bclamp(agent.constraints.bnd["vm-bnd"], vm_pred)
#                 va_pred = bclamp(agent.constraints.bnd["va-bnd"], va_pred)
#                 pg_pred = bclamp(agent.constraints.bnd["pg-bnd"], pg_pred)

#                 ovm, ova, opg = ivm + Δ_vm, iva + Δ_va, ipg + Δ_pg
#                 oflow_pred = get_Flows(agent.constraints.olc, vm_pred, va_pred)

#                 (_, pd_o, _, _) = get_components_from_Sd(sd, agent.datainfo)
#                 (pf_pred, _, pt_pred, _) = get_components_from_Sij(oflow_pred, agent.datainfo)
#                 klc_loss = get_losses(agent.constraints.pbc, pd_o, pg_pred, vm_pred, pf_pred, pt_pred)

#                 push!(err["vm"], F.l1_loss(vm_pred, ovm).item())
#                 push!(err["va"], F.l1_loss(va_pred, ova).item())
#                 push!(err["pg"], F.l1_loss(pg_pred, opg).item())
#                 push!(err["ohm"], F.l1_loss(oflow_pred, oflow).item())
#                 push!(err["klc"], klc_loss.item())

#                 # Save output
#                 merge!(results, collect_results(sd, ovm, ova, opg, oflow,
#                                                va_pred, vm_pred, pg_pred, oflow_pred,
#                                                agent.datainfo))
#                next!(p)
#             end
#             GC.gc(false)
#             println("$(@sprintf("Test Errors: \tvm: %.6f \tva: %.6f \tpg: %.6f \tflow: %.6f \tflow-balance: %.6f",
#                                 mean(err["vm"]), mean(err["va"]), mean(err["pg"]), mean(err["ohm"]), mean(err["klc"]) ))")
#         end
#     end
#     return results, err
# end

# function collect_results(sd, vm, va, pg, flow, va_pred, vm_pred, pg_pred, flow_pred, dinfo)
#     #pd_true, qd_true = get_Sd_from_pred(sd, dinfo, true)
#     pd_old, pd_true, qd_old, qd_true = get_components_from_Sd(sd, dinfo, true)

#     vm_pred, va_pred, pg_pred = toarray(vm_pred), toarray(va_pred), toarray(pg_pred)
#     vm_true, va_true, pg_true = toarray(vm), toarray(va), toarray(pg)
#     pf_pred, qf_pred, pt_pred, qt_pred = get_flows(flow_pred, true)
#     pf_true, qf_true, pt_true, qt_true = get_flows(flow, true)

#     batchsize = min(size(sd)[1])
#     results = Dict()
#     for i in 1:batchsize
#         test_index = agent.test_loader.current_indices[i]
#         results[test_index] = Dict(
#             "pd" => Dict(dinfo.keys["pd"][k] => v  for (k,v) in enumerate(pd_true[i,:])),
#             "qd" => Dict(dinfo.keys["qd"][k] => v  for (k,v) in enumerate(qd_true[i,:])),
#             "vm" => Dict(dinfo.keys["vm"][k] => v  for (k,v) in enumerate(vm_true[i,:])),
#             "va" => Dict(dinfo.keys["va"][k] => v  for (k,v) in enumerate(va_true[i,:])),
#             "pg" => Dict(dinfo.keys["pg"][k] => v  for (k,v) in enumerate(pg_true[i,:])),
#             "pf" => Dict(dinfo.keys["pf"][k] => v  for (k,v) in enumerate(pf_true[i,:])),
#             "qf" => Dict(dinfo.keys["qf"][k] => v  for (k,v) in enumerate(qf_true[i,:])),
#             "pt" => Dict(dinfo.keys["pt"][k] => v  for (k,v) in enumerate(pt_true[i,:])),
#             "qt" => Dict(dinfo.keys["qt"][k] => v  for (k,v) in enumerate(qt_true[i,:])),
#             "pred-vm" => Dict(dinfo.keys["vm"][k] => v  for (k,v) in enumerate(vm_pred[i,:])),
#             "pred-va" => Dict(dinfo.keys["va"][k] => v  for (k,v) in enumerate(va_pred[i,:])),
#             "pred-pg" => Dict(dinfo.keys["pg"][k] => v  for (k,v) in enumerate(pg_pred[i,:])),
#             "pred-pf" => Dict(dinfo.keys["pf"][k] => v  for (k,v) in enumerate(pf_pred[i,:])),
#             "pred-qf" => Dict(dinfo.keys["qf"][k] => v  for (k,v) in enumerate(qf_pred[i,:])),
#             "pred-pt" => Dict(dinfo.keys["pt"][k] => v  for (k,v) in enumerate(pt_pred[i,:])),
#             "pred-qt" => Dict(dinfo.keys["qt"][k] => v  for (k,v) in enumerate(qt_pred[i,:]))
#             )
#     end
#     return results
# end

function test_pf(agent::OpfAgent; dataset_json_path::AbstractString,
                 py_src_root::AbstractString=".", verbose::Bool=false)

        # --- helpers ---------------------------------------------------------
    mse_from_arrays(a, b) = mean((a .- b).^2)

    # convert (possibly PyTorch tensors via Ten(...)) to Julia Array on the right device
    toA(x, device) = toarray(Ten(x).to(device))

    # Try to run the model to get predictions; adapt to agent shape
    function _infer(agent, sd_t)
        if hasproperty(agent, :model)
            return agent.model(sd_t)
        elseif hasproperty(agent, :policy)
            return agent.policy(sd_t)
        else
            # Fallback: try call-overload
            return agent(sd_t)
        end
    end

    # Unpack model outputs into (vm, va, pg, qg) in a robust way
    function _unpack(out)
        # Dict or NamedTuple with common keys
        print(out)
        if out isa Dict
            vm = haskey(out, "vm") ? out["vm"] : out[:vm]
            va = haskey(out, "va") ? out["va"] : out[:va]
            pg = haskey(out, "pg") ? out["pg"] : out[:pg]
            #qg = haskey(out, "qg") ? out["qg"] : out[:qg]
            return vm, va, pg
        elseif out isa NamedTuple
            vm = haskey(out, :vm) ? out.vm : out[:vm]
            va = haskey(out, :va) ? out.va : out[:va]
            pg = haskey(out, :pg) ? out.pg : out[:pg]
            #qg = haskey(out, :qg) ? out.qg : out[:qg]
            return vm, va, pg 
        elseif out isa Tuple
            # order (vm, va, pg, qg) if tuple
            @assert length(out) ≥ 3 "Expected at least 3 outputs (vm, va, p)"
            return out[1], out[2], out[3]
        else
            error("Unrecognized model output type: $(typeof(out))")
        end
    end

    # --- accumulators (sum of squared errors and counts) -----------------
    sse = Dict("flow"=>0.0, "vm"=>0.0, "va"=>0.0, "pg"=>0.0)
    cnt = Dict("flow"=>0,    "vm"=>0,    "va"=>0,    "pg"=>0)

    # --- evaluation loop -------------------------------------------------
    for (sd, flow_gt, vm_gt, va_gt, pg_gt, qg_gt) in agent.test_loader
        # Tensors on device
        sd_t   = Ten(sd).to(agent.device)
        vm_gtA = toA(vm_gt, agent.device)
        va_gtA = toA(va_gt, agent.device)
        pg_gtA = toA(pg_gt, agent.device)
        #qg_gtA = toA(qg_gt, agent.device)
        flow_gtA = toA(flow_gt, agent.device)

        # Predictions
        out = _infer(agent, sd_t)
        vm_pred_t, va_pred_t, pg_pred_t = _unpack(out)

        vm_predA = toA(vm_pred_t, agent.device)
        va_predA = toA(va_pred_t, agent.device)
        pg_predA = toA(pg_pred_t, agent.device)
        #qg_predA = toA(qg_pred_t, agent.device)

        # Flow from predicted vm/va via Ohm's Law constraint  
        flow_pred_t = get_Flows(agent.constraints.olc, Ten(vm_pred_t).to(agent.device), Ten(va_pred_t).to(agent.device))
        flow_predA  = toA(flow_pred_t, agent.device)
        #println("*****")
        #println(pg_gtA)
        #println(pg_predA)
        #println("*****")
        # --- per-batch SSE updates (weighted properly over elements) -----
        sse["flow"] += sum((flow_predA .- flow_gtA).^2);  cnt["flow"] += length(flow_gtA)
        sse["vm"]   += sum((vm_predA   .- vm_gtA).^2);    cnt["vm"]   += length(vm_gtA)
        sse["va"]   += sum((va_predA   .- va_gtA).^2);    cnt["va"]   += length(va_gtA)
        sse["pg"]   += sum((pg_predA   .- pg_gtA).^2);    cnt["pg"]   += length(pg_gtA)
        #sse["qg"]   += sum((qg_predA   .- qg_gtA).^2);    cnt["qg"]   += length(qg_gtA)

        # (Optional) print per-batch MSEs
        # println("batch mse: flow=$(sse["flow"]/cnt["flow"]), vm=$(sse["vm"]/cnt["vm"]), va=$(sse["va"]/cnt["va"]), pg=$(sse["pg"]/cnt["pg"]), qg=$(sse["qg"]/cnt["qg"])")
    end

    # --- final MSEs over the entire test set -----------------------------
    mse_flow = sse["flow"] / max(cnt["flow"], 1)
    mse_vm   = sse["vm"]   / max(cnt["vm"],   1)
    mse_va   = sse["va"]   / max(cnt["va"],   1)
    mse_pg   = sse["pg"]   / max(cnt["pg"],   1)
    #mse_qg   = sse["qg"]   / max(cnt["qg"],   1)

    println("MSE over test set:")
    println("  flow: ", mse_flow)
    println("  vm  : ", mse_vm)
    println("  va  : ", mse_va)
    println("  pg  : ", mse_pg)
    #println("  qg  : ", mse_qg)

    
    ##############MSE ENDS HERE
                 # --- Make src.* importable ---
    sys = pyimport("sys")
    if !(abspath(py_src_root) in collect(sys."path"))
        push!(sys."path", abspath(py_src_root))
    end

    # --- Python deps ---
    torch  = pyimport("torch")
    ge     = pyimport("src.global_evaluation")
    gvars  = pyimport("src.global_vars")

    # --- Load dataset JSON ---
    ds        = JSON.parsefile(String(dataset_json_path))
    meta      = ds["meta"]
    net_paths = haskey(ds, "net_paths") ? ds["net_paths"] : meta["net_paths"]

    # --- Python helpers for PFEvaluator ---
    py"""
    import numpy as np, torch

    def _denorm_fn(y_norm, meta, device):
        #  model already outputs raw vm [pu], va [rad], pg [MW]
        return y_norm

    def _apply_pred_fn(base_net, meta, x_i, y_pred_i, denorm_fn, device):
        # y_pred_i must be (vm_pred, va_pred, pg_pred)
        if isinstance(y_pred_i, (tuple, list)):
            vm_t, va_t, pg_t = y_pred_i
        else:
            raise TypeError("Expected y_pred_i as (vm, va, pg)")
        vm = vm_t.detach().cpu().numpy().astype(float).ravel()
        va = va_t.detach().cpu().numpy().astype(float).ravel()
        pg = pg_t.detach().cpu().numpy().astype(float).ravel()

        bus_order = list(meta["bus_index_order"])
        gen_order = list(meta["gen_index_order"])

        # ext_grid setpoints
        if hasattr(base_net, "ext_grid") and len(base_net.ext_grid):
            for idx, row in base_net.ext_grid.iterrows():
                b = int(row["bus"])
                if b in bus_order:
                    j = bus_order.index(b)
                    base_net.ext_grid.at[idx, "vm_pu"] = float(vm[j])
                    if "va_degree" in base_net.ext_grid.columns:
                        base_net.ext_grid.at[idx, "va_degree"] = float(np.degrees(va[j]))

        # PV bus vm setpoints (gens)
        if hasattr(base_net, "gen") and len(base_net.gen):
            for gi, row in base_net.gen.iterrows():
                b = int(row["bus"])
                if b in bus_order:
                    j = bus_order.index(b)
                    base_net.gen.at[gi, "vm_pu"] = float(vm[j])

        # generator active powers in gen_order
        if hasattr(base_net, "gen") and len(base_net.gen):
            for k, gi in enumerate(gen_order):
                if gi < len(base_net.gen):
                    base_net.gen.at[int(gi), "p_mw"] = float(pg[k])

        return base_net

    def _append_pair(lst, a, b):
        lst.append((a, b))
    """

    # --- Build a Python loader [(xb, yb), ...] from the Julia loader ---
    # yb is unused when collect_norm_mse=False; we pass a dummy tensor = xb

    pylist = pybuiltin("list")
    loader = pylist()  # []

    for (b, (sd, _flow, _vm, _va, _pg, _qg)) in enumerate(agent.test_loader)
        # keep on CPU here; _infer_batched will move to `device`
        sd_t = Ten(sd).detach()
        push!(loader, (sd_t, sd_t))  # (xb, yb) ; yb is unused by evaluator
    end

    println("Built Python loader with ", Int(py"len"(loader)), " items.")

    nbatches = 0
    ntotal = 0
    for (sd, flow, vm, va, pg, qg) in agent.test_loader
        nbatches += 1
        ntotal += size(sd, 1)
    end
    println("Iterated $nbatches batches, covering $ntotal samples.")
    # Sanity check
    println("Built Python loader with ", Int(py"len"(loader)), " items.")
    # --- Create evaluator & run ---
    py"""
    import src.global_evaluation as ge
    import torch

    def _safe_is_torch(self, obj):
        try:
            mod = getattr(getattr(obj, '__class__', object), '__module__', '')
        except Exception:
            return False
        return (ge.torch is not None) and ('torch' in str(mod))

    ge.PFEvaluator._is_torch = _safe_is_torch


    def _infer_batched(model, xb, device):
        # move input to correct device
        if hasattr(xb, "to"):
            xb = xb.to(device)
        y = model(xb)
        # expect (vm, va, pg)
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
        py"_infer_batched",                 # <-- key change
        gvars.feas_tol, gvars.max_pen, gvars.ctol,
        runpp_kwargs
    )

    # device: PFEvaluator accepts a torch.device or a string; passing agent.device is fine
    # --- Create evaluator & run (unchanged above) ---
    pyres = evaluator.evaluate(agent.model, loader, meta, net_paths, agent.device,
                            false, verbose)

    # Convert Python -> Julia 
    costs = Float64.(pyres["costs"])
    feas = Bool.(pyres["hard_feasible_flags"])
    feas_rate = Float64(pyres["feasibility_rate_%"])
    
    avg_cost = isempty(costs) ? NaN : mean(costs)

    infer_time_per_sample = haskey(pyres, "infer_time_per_sample_s") ? Float64(pyres["infer_time_per_sample_s"]) : NaN
    infer_kept_count      = haskey(pyres, "infer_time_kept_n") ? Int(pyres["infer_time_kept_n"]) : 0

    println("RETURNING DICT...")
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