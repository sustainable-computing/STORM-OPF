############################ powerbalance_constraint.jl ############################
using JSON, JuMP, Ipopt, PowerModels
PowerModels.silence() # suppress PM warnings/info
include("torch.jl")   

# Bring common PM helpers into scope and add tiny compat shims
using PowerModels: ref, ids, nw_ids
const _FORM = isdefined(PowerModels, :build_opf) ? PowerModels.build_opf : PowerModels.post_opf
const _INST = isdefined(PowerModels, :instantiate_model) ? PowerModels.instantiate_model : PowerModels.build_model

"""
    PowerBalanceConstraint

Encodes the bus power balance using model's predicted tensors:
    - pg, qg, pd, qd per-element vectors
    - vm per-bus voltage magnitude
    - pf/pt per-branch flow (from/to)

Checks (per bus i):
    ∑_g∈i pg - ∑_l∈i pd - Gs_i * Vm_i^2 ≈ ∑_br∈i (pf_i + pt_i)
"""
mutable struct PowerBalanceConstraint
    device::Any
    nbuses::Any
    bus_pg_idx::Any
    bus_mask_pg::Any
    bus_pd_idx::Any
    bus_mask_pd::Any
    bus_vm_idx::Any
    bus_gs_vals::Any
    f_buses_idx::Any
    t_buses_idx::Any
end

# --- helpers ---

# Natural-sort JSON flow tags and map each JSON position -> MATPOWER branch row key
# Assumes MATPOWER branch table is [all lines first, then all trafos], and JSON tags are "L<i>" then "T<i>"
function _row_for_pf_position(pnet::Dict, pf_tags::Vector{Any})
    # all branch row keys "1","2",... as strings in numeric order
    branch_rows = sort(collect(keys(pnet["branch"])); by = x -> parse(Int, String(x)))
    n = length(branch_rows)

    # how many transformer tags in JSON?
    Ttags = [t for t in pf_tags if startswith(t, "T")]
    nT    = length(Ttags)
    nL    = n - nT
    @assert nL >= 0 "Branch rows < #T tags — unexpected."

    line_rows  = branch_rows[1:nL]
    trafo_rows = branch_rows[(nL+1):n]

    # For each JSON position pos, return the corresponding MATPOWER branch row key
    row_for_pos = Vector{String}(undef, length(pf_tags))
    for (pos, tag) in enumerate(pf_tags)
        @assert (startswith(tag, "L") || startswith(tag, "T")) "Unknown branch tag: $tag"
        idx = try
            parse(Int, tag[2:end])     # "L10" -> 10
        catch
            error("Non-numeric branch tag index: $tag")
        end
        if startswith(tag, "L")
            @assert 0 <= idx < length(line_rows) "L-index $idx out of range 0..$(length(line_rows)-1)"
            row_for_pos[pos] = line_rows[idx + 1]
        else
            @assert 0 <= idx < length(trafo_rows) "T-index $idx out of range 0..$(length(trafo_rows)-1)"
            row_for_pos[pos] = trafo_rows[idx + 1]
        end
    end
    return row_for_pos
end

# Sum shunt parameter at a bus; type must be "gs" or "bs"
function get_bus_shunt(pm, bus_idx::Int, type::String)
    @assert type == "gs" || type == "bs"
    nw = first(nw_ids(pm))

    # prefer bus record
    b = ref(pm, nw, :bus, bus_idx)
    if haskey(b, type)
        return Float64(b[type])
    end

    # fallback: sum explicit shunts (guard if :shunt is absent)
    val = 0.0
    try
        for j in ids(pm, nw, :shunt)
            sd = ref(pm, nw, :shunt, j)
            if get(sd, "bus", -1) == bus_idx
                val += Float64(get(sd, type, 0.0))
            end
        end
    catch
        # keep 0.0
    end
    return val
end

# Robustly find the vm/va index whether JSON used 0- or 1-based bus keys
function get_Vol_bus_idx(pm, bus_idx, datainfo, pred_type)
    @assert pred_type in ["vm", "va"]
    idx = get_keypos(datainfo, pred_type, string(bus_idx))
    if idx === nothing
        idx = get_keypos(datainfo, pred_type, string(bus_idx - 1))
    end
    return idx === nothing ? -1 : idx
end

# Return vector of positions in the pg/pd vectors for all gens/loads at a bus.
# Adds 0/1-based fallback like get_Vol_bus_idx.
function get_Pow_bus_idx(pm, bus_idx, datainfo, pred_type)
    @assert pred_type in ["pg", "qg", "pd", "qd"]
    nw = first(nw_ids(pm))
    pm_type = pred_type in ["pg", "qg"] ? :bus_gens : :bus_loads
    elem_ids = ref(pm, nw, pm_type, bus_idx)           # Vector{Int}; may be empty
    if isempty(elem_ids)
        return [-1]
    end
    idxs = Int[]
    for e in elem_ids
        idx = get_keypos(datainfo, pred_type, string(e))
        if idx === nothing
            idx = get_keypos(datainfo, pred_type, string(e - 1))
        end
        if idx !== nothing
            push!(idxs, idx)
        end
    end
    return isempty(idxs) ? [-1] : idxs
end

# --------------------------- constructor ---------------------------

function PowerBalanceConstraint(datainfo, config, device=nothing)
    device === nothing && (device = torch.device("cpu"))
    BS   = config["batchsize"]
    pnet = datainfo.powernet

    # Build a PM model (compat with PM 0.13.x and newer)
    pm = _INST(pnet, ACPPowerModel, _FORM)
    nw = first(nw_ids(pm))

    # --- Buses ---
    bus_IDs = sort(collect(ids(pm, nw, :bus)))  # Vector{Int}
    N = length(bus_IDs)

    # --- Generator/load index lists per bus (into the pg/pd vectors) ---
    bus_pg_idx = [get_Pow_bus_idx(pm, bus_i, datainfo, "pg") for bus_i in bus_IDs]
    mask_pg    = [idxs[1] == -1 ? 0.0 : 1.0 for idxs in bus_pg_idx]
    bus_pg_idx = [idxs[1] == -1 ? iTen([0]).to(device) : iTen(idxs .- 1).to(device) for idxs in bus_pg_idx]
    #mask_pg    = Ten(repeat([mask_pg]; outer=BS)).to(device)
    mask_pg_vec = Float32.(mask_pg)
    mask_pg_mat = repeat(reshape(mask_pg_vec, 1, :), BS, 1)
    mask_pg     = Ten(mask_pg_mat).to(device)

    bus_pd_idx = [get_Pow_bus_idx(pm, bus_i, datainfo, "pd") for bus_i in bus_IDs]
    mask_pd    = [idxs[1] == -1 ? 0.0 : 1.0 for idxs in bus_pd_idx]
    bus_pd_idx = [idxs[1] == -1 ? iTen([0]).to(device) : iTen(idxs .- 1).to(device) for idxs in bus_pd_idx]
    #mask_pd    = Ten(repeat([mask_pd]; outer=BS)).to(device)
    mask_pd_vec = Float32.(mask_pd)
    mask_pd_mat = repeat(reshape(mask_pd_vec, 1, :), BS, 1)
    mask_pd     = Ten(mask_pd_mat).to(device)

    # --- Shunt Gs per bus (sum of all shunts at that bus) ---
    bus_gs = [get_bus_shunt(pm, bus_i, "gs") for bus_i in bus_IDs]
    #bus_gs = Ten(repeat([bus_gs]; outer=BS)).to(device)
    bus_gs_vec  = Float32.(bus_gs)
    bus_gs_mat  = repeat(reshape(bus_gs_vec, 1, :), BS, 1)
    bus_gs      = Ten(bus_gs_mat).to(device)

    # --- vm index per bus (robust to 0/1-based JSON keys) ---
    bus_vm_idx = [get_Vol_bus_idx(pm, bus_i, datainfo, "vm") for bus_i in bus_IDs]
    bus_vm_idx = iTen(bus_vm_idx .- 1).to(device)

    # --- Map pf/pt JSON positions -> MATPOWER branch rows, then to bus incidence ---
    pf_tags = collect(datainfo.keys["pf"])            # canonical JSON order
    row_for_pos = _row_for_pf_position(pnet, pf_tags) # JSON pos -> MATPOWER row key
    nb = length(pf_tags)

    # For each bus, which JSON positions correspond to from/to incidence?
    f_pos = Vector{Vector{Int}}(undef, N)
    t_pos = Vector{Vector{Int}}(undef, N)
    for (k, bus_i) in enumerate(bus_IDs)
        f_here = Int[]
        t_here = Int[]
        for pos in 1:nb
            brow = row_for_pos[pos]
            br   = pnet["branch"][brow]
            if br["f_bus"] == bus_i
                push!(f_here, pos)
            end
            if br["t_bus"] == bus_i
                push!(t_here, pos)
            end
        end
        f_pos[k] = f_here
        t_pos[k] = t_here
    end

    # tensors of 0-based indices for torch.index_select
    f_buses_idx = [ iTen((pos .- 1)).to(device) for pos in f_pos ]
    t_buses_idx = [ iTen((pos .- 1)).to(device) for pos in t_pos ]

    return PowerBalanceConstraint(
        device, N,
        bus_pg_idx, mask_pg,
        bus_pd_idx, mask_pd,
        bus_vm_idx, bus_gs,
        f_buses_idx, t_buses_idx
    )
end

# --------------------------- loss ---------------------------

function get_losses(pbc::PowerBalanceConstraint, pd, pg, vm, pf, pt)
    N = pbc.nbuses

    # infer current batch size from the tensors
    curr_BS = Int(pg.shape[1])
    mask_BS = Int(pbc.bus_mask_pg.shape[1])

    # narrow masks/constants only if needed
    mask_pg = (curr_BS != mask_BS) ? pbc.bus_mask_pg.narrow(0, 0, curr_BS) : pbc.bus_mask_pg
    mask_pd = (curr_BS != mask_BS) ? pbc.bus_mask_pd.narrow(0, 0, curr_BS) : pbc.bus_mask_pd
    Gs      = (curr_BS != mask_BS) ? pbc.bus_gs_vals.narrow(0, 0, curr_BS) : pbc.bus_gs_vals

    # Build per-bus sums (stay in torch the whole time)
    Pg = torch.stack([torch.sum(torch.index_select(pg, 1, pbc.bus_pg_idx[i]), dim=1) for i in 1:N]).t()
    Pd = torch.stack([torch.sum(torch.index_select(pd, 1, pbc.bus_pd_idx[i]), dim=1) for i in 1:N]).t()
    Ff = torch.stack([torch.sum(torch.index_select(pf, 1, pbc.f_buses_idx[i]), dim=1) for i in 1:N]).t()
    Ft = torch.stack([torch.sum(torch.index_select(pt, 1, pbc.t_buses_idx[i]), dim=1) for i in 1:N]).t()
    Vm = torch.index_select(vm, 1, pbc.bus_vm_idx)

    # apply masks with torch ops
    Pg = torch.mul(Pg, mask_pg)
    Pd = torch.mul(Pd, mask_pd)

    # all math in torch (no Julia broadcasting)
    Vm2 = torch.pow(Vm, 2)
    lhs = torch.add(Ff, Ft)
    rhs = torch.sub(torch.sub(Pg, Pd), torch.mul(Gs, Vm2))

    return F.l1_loss(lhs, rhs)
end
