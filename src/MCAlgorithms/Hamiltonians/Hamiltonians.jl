using MacroTools
RuntimeGeneratedFunctions.init(@__MODULE__)
include("Ising.jl")
include("GaussianBernoulli.jl")
include("Clamping.jl")

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

params(::Type{T}, g) where T <: Hamiltonian = params(T, eltype(g))

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
function Hamiltonian_Builder(::Type{Algo}, graph, hamiltonians::Type{<:Hamiltonian}...) where {Algo <: MCAlgorithm}
    addparams!(graph, merge(params.(hamiltonians, eltype(graph))...))
    required_H = requires(Algo)
    H_ex = H_expr(required_H, graph, hamiltonians...)
    @RuntimeGeneratedFunction(H_ex)
end
export Hamiltonian_Builder

# function hamiltonian_ij_expr(graph, exprs...)
    
# end

