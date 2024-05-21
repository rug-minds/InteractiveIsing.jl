using MacroTools
RuntimeGeneratedFunctions.init(@__MODULE__)
include("Ising.jl")
include("GaussianBernoulli.jl")
include("Clamping.jl")

export CompositeHamiltonian
"""
Type for a composition of Hamiltonians
"""
struct CompositeHamiltonian{T} <: Hamiltonian end
CompositeHamiltonian(hamiltonians::Type{<:Hamiltonian}...) = CompositeHamiltonian{Tuple{hamiltonians...}}
function Base.iterate(c::Type{<:Hamiltonian}, state = 1)
    if state == 1
        return c, 2
    else
        return nothing
    end
end
Base.iterate(c::Type{<:CompositeHamiltonian}, state = 1) = iterate(c.parameters[1].parameters, state)
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

params(::Type{T}, g::IsingGraph) where T <: Hamiltonian = params(T, eltype(g))
params(c::Type{<:CompositeHamiltonian}, graph::IsingGraph) = merge(params.(hamiltonians(c), eltype(graph))...)

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
Doesn't work if vector param is written as v_i
"""
function Hamiltonian_Builder(::Type{Algo}, graph, hamiltonians::Type{<:Hamiltonian}) where {Algo <: MCAlgorithm}
    addparams!(graph, params(hamiltonians, graph))
    required_H = requires(Algo)
    H_ex = H_expr(required_H, graph, hamiltonians...)
    # Replace the symbols that are inactive by their default values
    H_ex = replace_inactive_symbs(graph.params, Meta.parse(H_ex))
    # Replace the symbols that are reserved by the algorithm
    H_ex = replace_reserved(Metropolis, H_ex)
    H_ex = replace_indices(H_ex)
    H_ex = replace_params(graph.params, H_ex)

    @RuntimeGeneratedFunction(H_ex)
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

# function hamiltonian_ij_expr(graph, exprs...)
    
# end

