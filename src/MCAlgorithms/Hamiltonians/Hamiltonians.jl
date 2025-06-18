using MacroTools
RuntimeGeneratedFunctions.init(@__MODULE__)

include("HamiltonianTerms.jl")

export CompositeHamiltonian, HamiltonianTerms


## OLD
"""
Type for a composition of Hamiltonians
"""
struct CompositeHamiltonian{T} <: Hamiltonian end

CompositeHamiltonian(hamiltonians::Hamiltonian...) = CompositeHamiltonian{tuple(hamiltonians...)}

Base.length(h::Hamiltonian) = 1
Base.length(c::CompositeHamiltonian) = length(c.parameters)
Base.length(h::Type{<:Hamiltonian}) = 1
Base.length(c::Type{<:CompositeHamiltonian}) = length(c.parameters)
Base.iterate(h::Hamiltonian, state = 1) = state == 1 ? (h, 2) : nothing
Base.iterate(c::CompositeHamiltonian, state = 1) = state <= length(c.parameters[1]) ? c.parameters[state] : nothing
Base.iterate(ht::Type{H}, state = 1) where H <: Hamiltonian = iterate(H(), state)
Base.iterate(c::Type{CH}, state = 1) where CH <: CompositeHamiltonian = iterate(CH(), state) 
function Base.getindex(h::Hamiltonian, i::Int)
    @assert i == 1
    return h
end
Base.getindex(c::CompositeHamiltonian, i::Int) = c.parameters[i]
Base.broadcastable(c::CompositeHamiltonian{Hs}) where {Hs} = Hs


include("Linear.jl")
include("MagField.jl")
include("Ising.jl")
include("IsingOLD.jl")
include("GaussianBernoulli.jl")
include("Clamping.jl")
include("DepolarisationField.jl")


hamiltonians(c::Type{<:CompositeHamiltonian}) = (c.parameters[1].parameters...,)


"""
Struct containing expressions and a set of not unique but sorted indexes that are used in the expressions
This way algorithms can just use only the types of indexes that they can use
"""
struct HamiltonianExprs
    exprs::Dict{Vector{Symbol},Expr}
end

struct HExpression
    collect_expr::Expr
    return_expr::Expr
end


struct HamiltonianParams
    names::Vector{Symbol}
    types::Vector{DataType}
    defaultvals::Vector{Real}
    descriptions::Vector{String}
end

function GatherHamiltonianParams(tups::Tuple{Symbol, DataType, <:Real, String}...)
    symbs = collect(map(tup->tup[1], tups))
    types = collect(map(tup->tup[2], tups))
    defaultvals = collect(map(tup->tup[3], tups))
    descriptions = collect(map(tup->tup[4], tups))
    return HamiltonianParams(symbs, types, defaultvals, descriptions)
end

params(c::Type{<:CompositeHamiltonian}, graph::IsingGraph) = merge(params.(hamiltonians(c), eltype(graph))...)
params(h::Type{H}, graph::IsingGraph) where H <: Hamiltonian = params(H(), eltype(graph))
params(h::Hamiltonian, g::IsingGraph) = params(h, eltype(g))

args(::Type{H}) where H <: Hamiltonian = args(H())

function merge(params::HamiltonianParams...) 
    emptyparams = HamiltonianParams(Symbol[], DataType[], Real[], String[])
    for p in params
        append!(emptyparams.names, p.names)
        append!(emptyparams.types, p.types)
        append!(emptyparams.defaultvals, p.defaultvals)
        append!(emptyparams.descriptions, p.descriptions)
    end
    return emptyparams
end


Base.setindex!(h::HamiltonianExprs, expr::Expr, terms::Vector{Symbol}) = h.exprs[sort(terms)] = expr
Base.getindex(h::HamiltonianExprs, terms::Vector{Symbol}) = h.exprs[sort(terms)]
Base.eachindex(h::HamiltonianExprs) = eachindex(h.exprs)
HamiltonianExprs(pairs::Pair{Vector{Symbol},Expr}...) = HamiltonianExprs(Dict(pairs...))
HamiltonianExprs(pair::Pair{Symbol,Expr}...) = HamiltonianExprs(Dict(map(pair->[first(pair)]=>last(pair),ps)...))

"""
Adds the paramvals to g.params, overwrites the old ones
"""
function addHparams(graph, oldparams, hamiltonian_params)
    pairs = Pair{Symbol, ParamVal}[]
    for index in eachindex(hamiltonian_params.names)
        type  = hamiltonian_params.types[index]
        val = nothing
        if type <: Vector
            val = zeros(eltype(type), length(graph.state))
        else
            val = zero(type)
        end
        push!(pairs, hamiltonian_params.names[index] => ParamVal(val, hamiltonian_params.defaultvals[index], hamiltonian_params.descriptions[index]))
    end
    params = Parameters(;pairs..., get_nt(oldparams)...)
end
"""
Doesn't work if vector param is written as v_i
"""
function Hamiltonian_Builder(::Type{algo}, graph, oldparams, hamiltonians::Hamiltonian) where {algo <: MCAlgorithm}
    gparams = addHparams(graph, oldparams, params(hamiltonians, graph))
    required_H = requires(algo)

    H_ex = H_expr(required_H, graph, hamiltonians...)
 
    H_ex = param_function(Meta.parse(H_ex), algo, gparams)

    (;ΔH = @RuntimeGeneratedFunction(H_ex), gparams)
end

rawH(algo, gr::IsingGraph) = Meta.parse(H_expr(requires(algo), gr, gr.hamiltonian...))
getH(gr::IsingGraph) = Hamiltonian_Builder(gr.default_algorithm, gr, gr.hamiltonian)
getH(algo, gr::IsingGraph) = Hamiltonian_Builder(algo, gr, gr.hamiltonian)

function setH!(gr, Hs...)
    if length(Hs) == 1
        return gr.hamiltonian = Hs[1]
    else
        gr.hamiltonian = CompositeHamiltonian(Hs...)
    end
end

export Hamiltonian_Builder, getH, rawH, setH!



### NEW SYSTEM

function dh(h::Hamiltonian, args; j)
    (;contractions, multiplications) = get_terms(h)
    return (args.newstate - args.gstate[j]) * contractions(args) + multiplications(args)
end