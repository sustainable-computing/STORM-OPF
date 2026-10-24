using Random: MersenneTwister,randperm!, shuffle!
using StatsBase

mutable struct DataLoader
    dataset::Tuple
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
    n::Int
    current_indices::Vector{Int}
end

# === NEW: single-loader constructor (no implicit split) ===
function DataLoader(dataset::NTuple{N,AbstractArray};
                    batchsize::Int=100,
                    shuffle::Bool=false,
                    indices::Union{Nothing,AbstractVector{Int}}=nothing
                    ) where N
    f = first.(size.(dataset))       # number of samples in each array
    seed= 98745#43#321   #using this seed to get a deterministic shuffle during training
    n = first(f)
    all(n .== f) || throw(DimensionMismatch("All arrays must share the same first dimension."))
    idx = indices === nothing ? collect(1:n) : collect(indices)

    it = DataLoader(dataset, batchsize, shuffle, idx, length(idx), Int[])
    # deterministic initial shuffle
    if it.shuffle
        rng = MersenneTwister(seed )
        shuffle!(rng, it.indices)
    end
    return it
    #shuffle && randperm!(idx)
    #return DataLoader(dataset, batchsize, shuffle, idx, length(idx), Int[])
end

# varargs convenience 
DataLoader(dataset...; kwargs...) = DataLoader(tuple(dataset...); kwargs...)

# === OPTIONALLy: we just keep the old split behavior under a new name ===
function split_loaders(dataset::NTuple{N,AbstractArray};
                       batchsize::Int=100,
                       shuffle::Bool=false,
                       train_split::Float64=0.8) where N
    f = first.(size.(dataset)); n = first(f)
    all(n .== f) || throw(DimensionMismatch("All arrays must share the same first dimension."))
    ntrain = Int(round(train_split * n))
    train_idx = sample(1:n, ntrain, replace=false)
    test_idx  = setdiff(1:n, train_idx)
    shuffle && randperm!(train_idx)
    return (
        DataLoader(dataset; batchsize=batchsize, shuffle=shuffle, indices=train_idx),
        DataLoader(dataset; batchsize=batchsize, shuffle=shuffle, indices=test_idx),
    )
end

# varargs convenience for split
split_loaders(dataset...; kwargs...) = split_loaders(tuple(dataset...); kwargs...)

function Base.iterate(it::DataLoader, start=1)
    if start > it.n
        it.shuffle && randperm!(it.indices)
        return nothing
    end
    nextstart = min(start + it.batchsize, it.n + 1)
    i = it.indices[start:nextstart-1]
    it.current_indices = i
    element = Tuple(copy(view(x, i, :)) for x in it.dataset)  #  (N, features)
    return element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = typeof(it.dataset)

function Base.show(io::IO, it::DataLoader)
    print(io, "DataLoader(dataset size = $(it.n), batchsize = $(it.batchsize), shuffle = $(it.shuffle))")
end
