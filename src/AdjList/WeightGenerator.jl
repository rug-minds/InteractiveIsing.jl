using BenchmarkTools, Distributions

struct WeightGenerator{Func <: Function , SelfFunc <: Union{Nothing, Function}, AddFunc <: Union{Nothing, Function}, MultFunc <: Union{Nothing, Function}}
    NN::Int16
    func::Func
    selfWeight::SelfFunc
    addDist::AddFunc
    multDist::MultFunc
    funcstr::Union{Nothing,String}
    selfstr::Union{Nothing,String}
    addstr::Union{Nothing,String}
    multstr::Union{Nothing,String}

    # function 

    function WeightGenerator(NN, func, selfWeight = nothing, addDist = nothing, multDist = nothing, funcstr = "", selfstr = "", addstr = "", multstr = "")
        new{typeof(func), typeof(selfWeight), typeof(addDist), typeof(multDist)}(NN, func, selfWeight, addDist, multDist, funcstr, selfstr, addstr, multstr)
    end

    function WeightGenerator(wg::WeightGenerator; NN = nothing, func = nothing, selfWeight = nothing, addDist = nothing, multDist = nothing, funcstr = nothing, selfstr = nothing, addstr = nothing, multstr = nothing)
        isnothing(NN) && (NN = wg.NN)
        isnothing(func) && (func = wg.func)
        isnothing(selfWeight) && (selfWeight = wg.selfWeight)
        isnothing(addDist) && (addDist = wg.addDist)
        isnothing(multDist) && (multDist = wg.multDist)
        isnothing(funcstr) && (funcstr = wg.funcstr)
        isnothing(selfstr) && (selfstr = wg.selfstr)
        isnothing(addstr) && (addstr = wg.addstr)
        isnothing(multstr) && (multstr = wg.multstr)

        new{typeof(func), typeof(selfWeight), typeof(addDist), typeof(multDist)}(NN, func, selfWeight, addDist, multDist, funcstr, selfstr, addstr, multstr)
    end
end

@setterGetter WeightGenerator
export WeightGenerator



function Base.show(io::IO, wg::WeightGenerator)
    println(io, "WeightGenerator with")
    println(io, "\t NN: \t\t\t\t", wg.NN)
    print(io, "\t func: \t\t\t\t", wg.funcstr)
    if wg.selfstr != nothing
        print(io, "\n\t Self Weight Function: \t\t", wg.selfstr)
    end
    if wg.addstr != nothing
        print(io, "\n\t Additive Distribution: \t", wg.addstr)
    end
    if wg.multstr != nothing
        print(io, "\n\t Multiplicative Distribution: \t", wg.multstr)
    end
end

"""
Gets the argument names for a method
"""
function method_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs]
end

"""
Gets the keywords args for a macro
"""
function prunekwargs(args...)
    @nospecialize
    firstarg = first(args)
    if isa(firstarg, Expr) && firstarg.head == :parameters
        return prunekwargs(BenchmarkTools.drop(args, 1)..., firstarg.args...)
    else
        params = collect(args)
        for ex in params
            if isa(ex, Expr) && ex.head == :(=)
                ex.head = :kw
            end
        end
        return params
    end
end

# Can this be turned into a function?
"""
Create a WeightGenerator
Accepted keywords are: NN, selfWeight, addDist, multDist, weightFunc
"""
macro WeightGenerator(wg_or_func, kwargs...)
    # Default Params
    NN = 1

    allowedargs_func = [:dr, :i, :j]
    allowedargs_self = [:i, :j]

    is_a_wg = false
    # When a string is given
    if isa(wg_or_func, String)
        # Set defaults
        selfWeight = addDist = multDist = nothing
        selfstr = addstr = multstr = nothing
        
        # Get the func str        
        funcstr = wg_or_func
        # Parse the func to get expression
        func = Meta.parse(funcstr)

        # MAIN FUNCTION
        # Get argument names
        argnames = method_argnames(last(methods(eval(func))))[2:end]
        # Check if argnames only contain a subset of the symbols :dr, :i, :j
        if !(all([arg ∈ allowedargs_func for arg in argnames]))
            error("Function must only contain arguments :dr, :i, :j")
        end

        # Get function body
        funcbody = func.args[2]
        func =  quote (;dr,i,j) -> Float32($funcbody) end
    else
        is_a_wg = true
        NN = func = selfWeight = addDist = multDist = funcstr = selfstr = addstr = multstr = nothing
    end

    # Get keyword arguments
    if !isempty(kwargs)
        params = prunekwargs(kwargs...)
        func_key = selfWeight_key = addDist_key = multDist_key = nothing
        # If keyword arguments get them
        for exp in params
            args = exp.args
            symb = args[1]
            val = args[2]
            if symb == :NN
                NN = val
            elseif symb == :selfWeight
                selfWeight_key = val
            elseif symb == :addDist
                addDist_key = val
            elseif symb == :multDist
                multDist_key = val
            elseif symb == :weightFunc
                func_key = val
            else
                error("Unknown keyword argument $symb")
            end
        end
    else
        func_key = selfWeight_key = addDist_key = multDist_key = nothing
    end
    


    # FUNC
    if !isnothing(func_key)
        funcstr = func_key
        funcexpr = Meta.parse(func_key)
        # get func argnames
        argnames_func = method_argnames(last(methods(eval(funcexpr))))[2:end]
        # Check if argnames only contain a subset of the symbols :dr, :i, :j
        if !(all([arg ∈ allowedargs_func for arg in argnames_func]))
            error("Function must only contain arguments :dr, :i, :j")
        end

        # Get function body
        funcbody = funcexpr.args[2]
        func = quote @inline (;dr,i,j) -> Float32($funcbody) end
    end
    
    # SELF WEIGHT
    if !isnothing(selfWeight_key)
        selfstr = selfWeight_key
        selfWeightExpr = Meta.parse(selfWeight_key)
        # println(selfWeightExpr)
        # println(methods(eval(selfWeightExpr)))
        # println(method_argnames(methods(selfWeightExpr)[]))

        # get selfweight argnames
        argnames_self = method_argnames(last(methods(eval(selfWeightExpr))))[2:end]
        # Check if argnames only contain a subset of the symbols :i, :j
        if !(all([arg ∈ allowedargs_self for arg in argnames_self]))
            error("Self weight Function must only contain arguments :i, :j")
        end
        
        # Get function body
        selfweightbody = selfWeightExpr.args[2]

        selfWeight = quote @inline (;i,j) -> Float32($selfweightbody) end
    end

    # ADDITIVE DISTRIBTION
    if !isnothing(addDist_key)
        addstr = addDist_key
        if contains(addDist_key, "rand") || contains(addDist_key, "sample")
            addDist = Meta.parse("@inline () -> Float32($addDist_key)")
        else
            addDist = Meta.parse("@inline () -> Float32(rand($addDist_key))")
        end
    end

    # MULTIPLICATIVE DISTRIBTION
    if !isnothing(multDist_key) 
        multstr = multDist_key
        if contains(multDist_key, "rand") || contains(multDist_key, "sample")
            multDist = Meta.parse("@inline () -> Float32($multDist_key)")
        else
            multDist = Meta.parse( "@inline () -> Float32(rand($multDist_key))" )
        end
    end
    if !is_a_wg
        return esc(quote
            WeightGenerator($NN, $func, $selfWeight, $addDist, $multDist, $funcstr, $selfstr, $addstr, $multstr)
        end)
    else
        return esc(quote
            WeightGenerator($wg_or_func, NN = $NN, func = $func, selfWeight = $selfWeight, addDist = $addDist, multDist = $multDist, funcstr = $funcstr, selfstr = $selfstr, addstr = $addstr, multstr = $multstr)
        end)
    end
end
macro WeightGenerator!(wg, args...) 
    return esc(quote
        $wg = @WeightGenerator $wg $(args...)
    end)
end
var"@WG" = var"@WeightGenerator"
var"@WG!" = var"@WeightGenerator!"
export @WeightGenerator, @WeightGenerator!, @WG, @WG!

@generated function getWeight(wg::WeightGenerator{Func, SelfFunc, AddFunc, MultFunc}, dr, i, j) where {Func, SelfFunc, AddFunc, MultFunc}
    return Meta.parse("wg.func(;dr,i,j)"*(!isa(MultFunc, Type{Nothing})*"*wg.multDist()" * (!isa(AddFunc, Type{Nothing})*" + wg.addDist()")))
end

@generated function getSelfWeight(wg::WeightGenerator{Func, SelfFunc, AddFunc, MultFunc}, i, j) where {Func, SelfFunc, AddFunc, MultFunc}
    return Meta.parse("wg.selfWeight(;i,j)"*(!isa(MultFunc, Type{Nothing})*"*wg.multDist()" * (!isa(AddFunc, Type{Nothing})*" + wg.addDist()")))
end
export getWeight, getSelfWeight

using StaticArrays
abstract type AbstractPreAlloc{T} <: AbstractVector{T} end

mutable struct Prealloc{T} <: AbstractPreAlloc{T}
    const vec::Vector{T} 
    used::Int32
    const maxsize::Int32

    function Prealloc(type::Type, N)
        vec = Vector{type}(undef, N)
        maxsize = N
        used = 0
        new{type}(vec, used, maxsize)
    end

end

# Using StaticArrays
mutable struct SPrealloc{S,T} <: AbstractPreAlloc{T}
    const vec::MVector{S,T} 
    used::Int32
    const maxsize::Int32

    function SPrealloc(type::Type, N)
        vec = MVector{N, type}(Vector{type}(undef, N))
        maxsize = N
        used = 0
        new{N, type}(vec, used, maxsize)
    end

end

struct ThreadedPrealloc{PreallocT}
    vec::Vector{PreallocT}
end
@inline getindex(pre::ThreadedPrealloc, i) = pre.vec[i]

function ThreadedPrealloc(type::Type, N, nthreads)
    PreallocT = typeof(SPrealloc(type, N))
    vec = Vector{PreallocT}(undef, nthreads)
    for i in 1:nthreads
        vec[i] = SPrealloc(type, N)
    end
    return ThreadedPrealloc(vec)
end

getindex(pre::AbstractPreAlloc, i) = pre.vec[i]
setindex!(pre::AbstractPreAlloc, tup, i) = (pre.vec[i] = tup; pre.used = max(pre.used, i))
length(pre::AbstractPreAlloc) = pre.used
push!(pre::AbstractPreAlloc, tup) = (pre.vec[pre.used+1] = tup; pre.used += 1)
reset!(pre::AbstractPreAlloc) = (pre.used = 0; return)
size(pre::AbstractPreAlloc) = tuple(pre.used)

export Prealloc, SPrealloc


Base.zero(::Type{NTuple{N,T}}) where {N,T} = NTuple{N,T}(Base.zero(T) for i in 1:N)
Base.zero(::Type{Tuple{T1,T2}}) where {T1,T2} = (Base.zero(T1), Base.zero(T2))

abstract type SelfType end
struct Self <: SelfType end
struct NoSelf <: SelfType end

function SelfType(wg::WeightGenerator{A,SelfFunc,B,C}) where {A,SelfFunc,B,C}
    if isa(SelfFunc, Type{Nothing})
        return NoSelf()
    else
        return Self()
    end
end
export SelfType

genAdj!(layer::L, wg::WG) where {L <: AbstractIsingLayer, WG <: WeightGenerator} = genAdj!(SelfType(wg), layer, top(layer), wg)

# Base.@propagate_inbounds function genAdj!(layer::L, layer_top::LT, wg::WG) where {L <: AbstractIsingLayer, LT, WG <: WeightGenerator}
Base.@propagate_inbounds function genAdj!(st::ST, layer::L, layer_top::LT, wg::WG) where {ST <: SelfType, L <: AbstractIsingLayer, LT, WG <: WeightGenerator}
    fNN = NN(wg)

    # Get the adjacency list
    fadj = adj(graph(layer))
    # Number of states
    num_verts = nStates(layer)

    # Keeps track of the last index that was accessed
    # Used to avoid searching every time in the adjacency list
    last_access = ones(Int32, num_verts)

    pre_3tuple = SPrealloc(NTuple{3, Int32}, (2*fNN+1)^2)

    # layer_top = top(layer)
    len = glength(layer)
    wid = gwidth(layer)

    for vert_idx in 1:num_verts
        vert_i, vert_j = idxToCoord(vert_idx, len)
        getConnIdxs!(st, vert_idx, vert_i, vert_j, len ,wid, fNN, pre_3tuple)
        fillEntry!(st, layer, vert_idx, vert_i, vert_j, layer_top, wg, pre_3tuple, last_access)
        reset!(pre_3tuple)
    end

    # For performance
    # Probably helps with data locality
    adj(graph(layer)) .= deepcopy(fadj)

    return fadj
end
export genAdj!


"""
Get all indices of a vertex with idx vert_idx and coordinates vert_i, vert_j
that are larger than vert_idx
Works in layer indices
"""
function getConnIdxs!(::NoSelf, vert_idx, vert_i, vert_j, len, wid, NN, pre_3tuple)
    for j in -NN:NN
        for i in -NN:NN
            (i == 0 && j == 0) && continue
            conn_i, conn_j = latmod(vert_i + i, vert_j + j, len, wid)
            conn_idx = coordToIdx(conn_i, conn_j, len)

            conn_idx < vert_idx && continue

            push!(pre_3tuple, (conn_i, conn_j, conn_idx))
        end
    end
end

"""
Get all indices of a vertex with idx vert_idx and coordinates vert_i, vert_j
that are larger than vert_idx and include self connection
"""
function getConnIdxs!(::Self, vert_idx, vert_i, vert_j, len, wid, NN, pre_3tuple)
    for j in -NN:NN
        for i in -NN:NN
            conn_i, conn_j = latmod(vert_i + i, vert_j + j, len, wid)
            conn_idx = coordToIdx(conn_i, conn_j, len)

            conn_idx < vert_idx && continue

            push!(pre_3tuple, (conn_i, conn_j, conn_idx))
        end
    end
end

function fillEntry!(::NoSelf, layer::L, idx, vert_i, vert_j, layer_top, wg, pre_3tuple, last_access) where L # Force specialization

    sort!(pre_3tuple, by = x -> x[3])
    for conn in pre_3tuple
        conn_i, conn_j = conn[1], conn[2]
        dr = dist(vert_i, vert_j, conn_i, conn_j, layer_top)
        weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2)
        if weight != 0
            lastacces1, lastacces2 = addWeight!(layer, idx, conn[3], weight, sidx1 = last_access[idx], sidx2 = last_access[conn[3]])
            last_access[idx] = lastacces1
            last_access[conn[3]] = lastacces2
        end
    end

    return
end

# Fill with self weights
function fillEntry!(::Self, layer::L, idx, vert_i, vert_j, layer_top, wg, pre_3tuple, last_access) where L # Force specialization

    sort!(pre_3tuple, by = x -> x[3])
    for conn in pre_3tuple
        if conn[3] == idx
            weight = getSelfWeight(wg, vert_i, vert_j)
        else
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, layer_top)
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2)
        end
        if weight != 0
            lastacces1, lastacces2 = addWeight!(layer, idx, conn[3], weight, sidx1 = last_access[idx], sidx2 = last_access[conn[3]])
            last_access[idx] = lastacces1
            last_access[conn[3]] = lastacces2
        end
    end
    
    return
end

const wg_isingdefault = @WeightGenerator "(dr) -> dr == 1" NN = 1
