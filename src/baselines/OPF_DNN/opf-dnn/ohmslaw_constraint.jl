# ohmslaw_constraint.jl
include("torch.jl")

# ───────────────────────── helpers ─────────────────────────

# Natural-sort flow tags: L0,L1,... then T0,T1,...
_sort_pf_tags(tags::AbstractVector{<:AbstractString}) = _sort_pf_tags(Any[String(t) for t in tags])
function _sort_pf_tags(pf_keys::Vector{Any})
    Ls = [t for t in pf_keys if startswith(t, "L")]
    Ts = [t for t in pf_keys if startswith(t, "T")]
    ln = sort(Ls; by = t -> parse(Int, t[2:end]))
    tn = sort(Ts; by = t -> parse(Int, t[2:end]))
    return vcat(ln, tn)
end

# Map a MATPOWER bus id (1..N) to position in vm_keys (works for 0- or 1-based keys).
function _make_bus_lookup(vm_keys::Vector{Any})
    parsed = map(k -> tryparse(Int, k), vm_keys)
    if all(!isnothing, parsed)
        vm_ints = Int.(parsed)
        inv = Dict{Int,Int}()
        for (pos, bid) in enumerate(vm_ints)
            inv[bid] = pos
        end
        if minimum(vm_ints) == 0
            return b::Int -> get(inv, b-1, 0)  # JSON keys 0-based
        else
            return b::Int -> get(inv, b, 0)    # JSON keys 1-based
        end
    else
        return b::Int -> begin
            pos = findfirst(==(string(b)), vm_keys)
            pos === nothing && (pos = findfirst(==(string(b-1)), vm_keys))
            pos === nothing ? 0 : pos
        end
    end
end

# All branch keys in numeric order
function _sorted_branch_rows(pnet::Dict)
    return sort(collect(keys(pnet["branch"])); by = x -> parse(Int, String(x)))
end

# Build tag -> row map by index:

function _build_tag2row_by_index(pnet::Dict, pf_tags::Vector{String})::Dict{String,String}
    ks  = _sorted_branch_rows(pnet)
    n   = length(ks)
    Tts = [t for t in pf_tags if startswith(t, "T")]
    nT  = length(Tts)
    nL  = n - nT
    nL >= 0 || error("Branch rows < #T tags — unexpected.")

    line_rows  = ks[1:nL]           # first nL rows are lines
    trafo_rows = ks[(nL+1):n]       # last nT rows are trafos

    tag2row = Dict{String,String}()
    for t in pf_tags
        startswith(t, "L") || startswith(t, "T") || error("Unknown branch tag: $t")
        idx = try
            parse(Int, t[2:end])    # e.g., "L10" -> 10  (0-based tag index)
        catch
            error("Non-numeric branch tag index: $t")
        end
        if startswith(t, "L")
            (0 <= idx < length(line_rows))  || error("L-index $idx out of range 0..$(length(line_rows)-1)")
            tag2row[t] = line_rows[idx+1]
        else
            (0 <= idx < length(trafo_rows)) || error("T-index $idx out of range 0..$(length(trafo_rows)-1)")
            tag2row[t] = trafo_rows[idx+1]
        end
    end
    return tag2row
end

# ───────────────────────── OhmsLawConstraint ─────────────────────────

mutable struct OhmsLawConstraint
    device::Any
    idx_bus_fr::Any
    idx_bus_to::Any
    tm::Any
    ta::Any
    g::Any
    b::Any
    p_fr1::Any
    q_fr1::Any
    p_to1::Any
    q_to1::Any
    Sbase::Float64           # scale pu -> MW/Mvar
end

function OhmsLawConstraint(datainfo, config, device=nothing)
    device === nothing && (device = torch.device("cpu"))
    pnet  = datainfo.powernet
    BS    = config["batchsize"]
    Sbase = Float64(get(pnet, "baseMVA", 100.0))   # from parsed MATPOWER

    # 1) Flow tags from JSON in a deterministic order matching the writer
    raw_tags = collect(datainfo.keys["pf"])
    br_tags  = _sort_pf_tags(raw_tags)

    # 2) Map tags -> MATPOWER row keys by index (lines first, then trafos)
    tag2row = _build_tag2row_by_index(pnet, br_tags)
    br_rows = [ tag2row[t] for t in br_tags ]  # preserve pf-tag order

    # 3) Gather parameters in that exact order
    fetch = r -> pnet["branch"][r]
    f_bus = [ fetch(r)["f_bus"] for r in br_rows ]'
    t_bus = [ fetch(r)["t_bus"] for r in br_rows ]'

    _gb   = [ calc_branch_y(fetch(r)) for r in br_rows ]
    g_vec = [ _gb[i][1] for i in 1:length(_gb) ]'
    b_vec = [ _gb[i][2] for i in 1:length(_gb) ]'

    g_frv = [ get(fetch(r), "g_fr", 0.0) for r in br_rows ]'
    b_frv = [ get(fetch(r), "b_fr", 0.0) for r in br_rows ]'
    g_tov = [ get(fetch(r), "g_to", 0.0) for r in br_rows ]'
    b_tov = [ get(fetch(r), "b_to", 0.0) for r in br_rows ]'

    tm_vec = [ let v = get(fetch(r), "tap", get(fetch(r), "ratio", 1.0))
                   v == 0.0 ? 1.0 : v
               end for r in br_rows ]'

    ta_vec = [ let v = get(fetch(r), "shift", get(fetch(r), "angle", 0.0))
                   abs(v) > π ? v * (π/180) : v   # convert deg->rad if it looks like degrees
               end for r in br_rows ]'

    # 4) Index vectors into vm/va using the provided bus-key set
    vm_keys = datainfo.keys["vm"]
    pos_of  = _make_bus_lookup(vm_keys)

    idx_fr = [begin
        b = Int(f_bus[i]); p = pos_of(b)
        p == 0 && error("Bus $b not found in vm_keys.")
        p
    end for i in 1:length(f_bus)]

    idx_to = [begin
        b = Int(t_bus[i]); p = pos_of(b)
        p == 0 && error("Bus $b not found in vm_keys.")
        p
    end for i in 1:length(t_bus)]

    # PyTorch indices must be 0-based
    idx_bus_fr = iTen(idx_fr .- 1).to(device)
    idx_bus_to = iTen(idx_to .- 1).to(device)

    # 5) Prebuild batched tensors in the exact (sorted) branch/tag order
    p_fr1 = Ten(repeat((g_vec .+ g_frv); outer=BS)).to(device)
    q_fr1 = Ten(repeat(-(b_vec .+ b_frv); outer=BS)).to(device)
    p_to1 = Ten(repeat((g_vec .+ g_tov); outer=BS)).to(device)
    q_to1 = Ten(repeat(-(b_vec .+ b_tov); outer=BS)).to(device)
    tmT   = Ten(repeat(tm_vec; outer=BS)).to(device)
    taT   = Ten(repeat(ta_vec; outer=BS)).to(device)
    gT    = Ten(repeat(g_vec;  outer=BS)).to(device)
    bT    = Ten(repeat(b_vec;  outer=BS)).to(device)

    return OhmsLawConstraint(device, idx_bus_fr, idx_bus_to, tmT, taT, gT, bT,
                             p_fr1, q_fr1, p_to1, q_to1, Sbase)
end

# Always slice the first BS rows of the prebuilt tensors
function get_batch(olc::OhmsLawConstraint, BS::Int)
    return  olc.tm.narrow(0, 0, BS),
            olc.ta.narrow(0, 0, BS),
            olc.g.narrow(0, 0, BS),
            olc.b.narrow(0, 0, BS),
            olc.p_fr1.narrow(0, 0, BS),
            olc.q_fr1.narrow(0, 0, BS),
            olc.p_to1.narrow(0, 0, BS),
            olc.q_to1.narrow(0, 0, BS)
end

function get_Flows(olc::OhmsLawConstraint, vm, va)
    vm_fr = torch.index_select(vm, 1, olc.idx_bus_fr)
    vm_to = torch.index_select(vm, 1, olc.idx_bus_to)
    va_fr = torch.index_select(va, 1, olc.idx_bus_fr)
    va_to = torch.index_select(va, 1, olc.idx_bus_to)

    BS = Int(vm.shape[1])
    tm, ta, g, b, p_fr1, q_fr1, p_to1, q_to1 = get_batch(olc, BS)

    # upstream side
    f2  = torch.pow(vm_fr / tm, 2.0)
    f3  = (vm_fr / tm) * vm_to
    _f3 = (va_fr - va_to - ta)
    f3c, f3s = torch.cos(_f3), torch.sin(_f3)
    p_fr = (p_fr1 * f2) - (g * f3 * f3c) - (b * f3 * f3s)
    q_fr = (q_fr1 * f2) + (b * f3 * f3c) - (g * f3 * f3s)  # matches sign/terms on upstream

    # downstream side
    t2  = torch.pow(vm_to, 2.0)
    t3  = vm_to * (vm_fr / tm)
    _t3 = (va_to - va_fr + ta)
    t3c, t3s = torch.cos(_t3), torch.sin(_t3)
    p_to = (p_to1 * t2) - (g * t3 * t3c) - (b * t3 * t3s)
    q_to = (q_to1 * t2) + (b * t3 * t3c) - (g * t3 * t3s)

    # Scale pu -> MW/Mvar to match JSON units
    flows_pu = torch.cat((p_fr, q_fr, p_to, q_to), 1)
    return olc.Sbase * flows_pu
end
