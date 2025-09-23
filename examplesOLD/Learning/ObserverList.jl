mutable struct SimpleAverage{T}
    sum::T
    count::Int
end
SimpleAverage(T) = SimpleAverage{T}(0,0)
Base.push!(avg::SimpleAverage, val) = (avg.sum += val; avg.count += 1)
avg(avg::SimpleAverage) = avg.sum/avg.count
reset!(avg::SimpleAverage) = (avg.sum = 0; avg.count = 0)

using SparseArrays
abstract type AbstractObserver{T} end

mutable struct WeightObserver{T, U, R} <: AbstractObserver{T}
    stateref::Vector{T}
    adjref::R
    adata::SimpleAverage{Float64}
    pair::Pair{Int32,Int32}
    nzidx1::Int32
    nzidx2::Int32
    val::T
    storage::U
end

WeightObserverType(sparse::SparseMatrixCSC{Tv,Ti}, storage::SType) where {Tv, Ti, SType} = 
    WeightObserver{Tv, SType, SparseMatrixCSC{Tv,Ti}}

WeightObserver(state, sparse::SparseMatrixCSC{Tv,Ti}, pair::Pair{Int32,Int32}, nzidx1::Int32, val::Tv, initstorage::U = 0.) where {Tv,Ti, U} = 
    WeightObserver{Tv, U, SparseMatrixCSC{Tv,Ti}}(state, sparse, SimpleAverage(Float64), pair, nzidx1, 0, val, initstorage)

idx1(ob::WeightObserver) = ob.pair.first
idx2(ob::WeightObserver) = ob.pair.second

function setval!(ob::WeightObserver, val)
    ob.adjref.nzval[ob.nzidx1] = val
    ob.adjref.nzval[ob.nzidx2] = val
    ob.val = val
end

setstorage!(ob::AbstractObserver, storage) = ob.storage = storage
addstorage!(ob::AbstractObserver, storage) = ob.storage += storage
subtractstorage!(ob::AbstractObserver, storage) = ob.storage -= storage
getstorage(ob::AbstractObserver) = ob.storage
resetstorage!(ob::AbstractObserver) = ob.storage = 0.
dividestorage!(ob::AbstractObserver, divisor) = ob.storage /= divisor
val(ob::WeightObserver) = ob.val
storage(ob::AbstractObserver) = ob.storage

resetavg!(ob::AbstractObserver) = reset!(ob.adata)

pushavg!(ob::WeightObserver, val) = push!(ob.adata, val)
function storage_to_val!(ob::AbstractObserver{T}, mult = one(T)) where {T}
    setval!(ob, convert(T,avg(ob.adata))*mult)
    resetstorage!(ob)
end

function add_storage_to_val!(ob::AbstractObserver{T}, mult = one(T)) where {T}
    setval!(ob, val(ob) + convert(T, avg(ob.adata))*mult)
    resetstorage!(ob)
end


struct WeightObservers{T, U, R}
    observers::Vector{WeightObserver{T, U, R}}
    adj::R
end

WeightObservers(sparse::SparseMatrixCSC{Tv,Ti}, storage = 0.) where {Tv,Ti} = Vector{WeightObserver{Tv, typeof(storage), SparseMatrixCSC{Tv,Ti}}}()

# Order pair so that the first index is always smaller than the second
function order_pair(pair::Pair{<:Integer,<:Integer})
    if pair.first < pair.second
        return pair
    else
        return Pair(pair.second, pair.first)
    end
end

# getall(obs::WeightObservers) = collect(values(obs.observers))

pair32(a,b) = Int32(a) => Int32(b)

Base.getindex(obs::WeightObservers, key) where {A,B} = getindex(obs.observers::Dict{A,B}, order_pair(key))::B
#Matrix acessing
Base.getindex(obs::WeightObservers, key1, key2) where {A,B} = getindex(obs.observers::Dict{A,B}, order_pair(key1=>key2))

Base.setindex!(obs::WeightObservers, val, key) = setindex!(obs.observers, val, order_pair(key))
Base.delete!(obs::WeightObservers, key) = deletekey!(obs.observers, order_pair(key))
Base.length(obs::WeightObservers) = length(obs.observers)
Base.sizehint!(obs::WeightObservers, sz) = sizehint!(obs.observers, sz)
Base.iterate(obs::WeightObservers, state=1) = iterate(obs.observers, state)

function setval!(obs::WeightObservers, val, pair)
    if haskey(obs, pair)
        setval!(obs[pair], val)
        return val
    end
    return val
end

function sparseobservers(g, adj::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    obs = WeightObservers(adj)
    sizehint!(obs, round(Int32,nnz(adj)/2))
    for col_idx in Int32.(1:size(adj,2))
        for ptr in nzrange(adj, col_idx)
            row_idx = adj.rowval[ptr]
            if haskey(obs, col_idx=>row_idx)
                # Add second index in an ordered way
                if ptr > obs[col_idx=>row_idx].nzidx1
                    obs[col_idx=>row_idx].nzidx2 = ptr
                else
                    obs[col_idx=>row_idx].nzidx2 = obs[col_idx=>row_idx].nzidx1
                    obs[col_idx=>row_idx].nzidx1 = ptr
                end
            else
                obs[col_idx=>row_idx] = WeightObserver(state(g), adj, col_idx=>row_idx, Int32(ptr), adj.nzval[ptr])
            end
        end
    end
    return obs
end

mutable struct BiasObserver{T, U} <: AbstractObserver{T}
    const stateref::Vector{T}
    const biasref::Vector{T}
    const adata::SimpleAverage{Float64}
    idx::Int32
    storage::U
end

idx(ob::BiasObserver) = ob.idx

BiasObserverType(state, storage::SType) where {SType} = 
    BiasObserver{eltype(state), SType}

setval!(ob::BiasObserver, val) = ob.biasref[ob.idx] = val
val(ob::BiasObserver) = ob.biasref[ob.idx]


pushavg!(ob::BiasObserver, val) = push!(ob.adata, val)

BiasObserver(state::Vector{Tv}, biasref::Vector{Tv}, idx::Int32, storage::SType = 1.) where {Tv, SType} = 
    BiasObserver{Tv, SType}(state, biasref, SimpleAverage(Float64), idx, storage)

struct BiasObservers{BType} <: AbstractVector{BType}
    observers::Vector{BType}
end
BiasObservers(state, storage = 0.) = let BType = BiasObserverType(state, storage)
    new{BType}(BType[]) end

getall(obs::BiasObservers) = collect(obs.observers)
Base.push!(obs::BiasObservers, ob) = push!(obs.observers, ob)
Base.size(obs::BiasObservers) = size(obs.observers)
Base.getindex(obs::BiasObservers, idx) = getindex(obs.observers, idx)
Base.setindex!(obs::BiasObservers, val, idx) = setindex!(obs.observers, val, idx)
Base.length(obs::BiasObservers) = length(obs.observers)
Base.eltype(obs::BiasObservers) = eltype(obs.observers)
Base.iterate(obs::BiasObservers, state=1) = iterate(obs.observers, state)



setval!(obs::BiasObservers, val, idx) = setval!(obs.observers[idx], val)

function biasobservers(g)
    obs = BiasObservers(BiasObserverType(state(g), 0.)[])
    for idx in Int32.(1:length(bfield(g)))
        push!(obs, BiasObserver(state(g), bfield(g), idx))
    end
    return obs
end

function pushavgstate!(g, obs::WeightObserver) 
    pushavg!(obs, - state(g)[idx1(obs)] * state(g)[idx2(obs)])
end

function pushavgstate!(g, obs::BiasObserver)
    pushavg!(obs, - state(g)[idx(obs)])
end