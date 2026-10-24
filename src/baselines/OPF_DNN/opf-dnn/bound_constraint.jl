include("torch.jl")

"""
    Struct to encode Bound constraints.
"""
mutable struct BoundConstraint
    lb::Any
    ub::Any
    zeros::Any
    is_abs::Bool

    function BoundConstraint(data, config, type, device=nothing)
        if device == nothing
            device = torch.device("cpu")
        end
        BS = config["batchsize"]

        @assert type in ["pg-bnd", "qg-bnd", "vg-bnd", "vm-bnd", "va-bnd", "flow-bnd"]
        _type = type[1:findfirst("-bnd", type)[1]-1]

        # ---- helpers to fetch canonical orders from JSON ----
        has_order = haskey(data, "order")
        get_order = (k) -> (has_order && haskey(data["order"], k)) ? data["order"][k] : nothing

        if _type == "flow"
            # rate_a must follow the SAME order as pf/qf/pt/qt ("pf" order)
            limdict = data["constraints"]["rate_a"]
            ord = get_order("pf")
            if ord === nothing
                # deterministic fallback if order not present
                ord = sort(collect(keys(limdict)))
            end
            # vector of Smax in the chosen order
            x_vec = [ limdict[k] for k in ord ]                      # length = n_flow
            x_lim = hcat(x_vec...)'                                  # (n_flow, 1)
            x_lim = x_lim.^2                                         # keep original semantics (S^2)
            x_min = nothing
            x_max = Ten(repeat(x_lim'; outer=BS)).to(device)         # (BS, n_flow)
            is_abs = true

        elseif _type == "va"
            # fixed angle bounds for ALL buses
            n_buses = length(data["experiments"][1]["va"])
            x_min = torch.zeros(BS, n_buses).fill_(-pi/6).to(device)
            x_max = torch.zeros(BS, n_buses).fill_( pi/6).to(device)
            is_abs = false

        else
            # pg/qg/vg/vm → read *_lim in the correct order
            # map vg to generator order (same indices as pg/qg)
            limkey = string(_type, "_lim")
            limdict = data["constraints"][limkey]

            ord = if _type == "vm"
                get_order("vm")
            elseif _type == "pg" || _type == "qg" || _type == "vg"
                get_order("pg")
            else
                nothing
            end
            if ord === nothing
                # deterministic fallback
                ord = sort(collect(keys(limdict)))
            end

            # each limdict[k] is a (min,max) pair
            x_lim = hcat([ limdict[k] for k in ord ]...)'            # (n, 2)
            x_min, x_max = selectdim(x_lim, 2, 1), selectdim(x_lim, 2, 2)  # (n,), (n,)
            x_min = Ten(repeat(x_min'; outer=BS)).to(device)         # (BS, n)
            x_max = Ten(repeat(x_max'; outer=BS)).to(device)         # (BS, n)
            is_abs = false
        end

        zeros = torch.zeros_like(x_max).to(device)
        new(x_min, x_max, zeros, is_abs)
    end
end

# Return bounds tensors aligned to x's (B, N) shape.
function get_bounds(bc::BoundConstraint, x)
    _lb, _ub, _z = bc.lb, bc.ub, bc.zeros

    # Use PyTorch sizes to disambiguate (x: B x N, ub: BS x N0)
    B  = Int(x.size(0))
    N  = Int(x.size(1))
    B0 = Int(bc.ub.size(0))
    N0 = Int(bc.ub.size(1))

    # Adjust batch (common on last, short batch)
    if B != B0
        _lb = bc.is_abs ? bc.lb : bc.lb.narrow(0, 0, B)
        _ub = bc.ub.narrow(0, 0, B)
        _z  = bc.zeros.narrow(0, 0, B)
    end
    # Adjust features (rare, but safer)
    if N != N0
        _lb = bc.is_abs ? _lb : _lb.narrow(1, 0, N)
        _ub = _ub.narrow(1, 0, N)
        _z  = _z.narrow(1, 0, N)
    end
    return _lb, _ub, _z
end

# Loss functions on a vector of variables bounds
function bound_penalty(bc::BoundConstraint, x)
    lb, ub, z = get_bounds(bc, x)
    if !bc.is_abs
        return torch.mean(torch.max(z, (lb - x))) +
               torch.mean(torch.max(z, (x - ub)))
    else
        return torch.mean(torch.max(z, (x - ub)))
    end
end

function bclamp(bc::BoundConstraint, x)
    lb, ub, z = get_bounds(bc, x)
    if !bc.is_abs
        return torch.max(torch.min(x, ub), lb)
    else
        return torch.min(x, ub)
    end
end
