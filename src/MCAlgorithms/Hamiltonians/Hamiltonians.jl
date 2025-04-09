using MacroTools
RuntimeGeneratedFunctions.init(@__MODULE__)

export CompositeHamiltonian, HamiltonianTerms
struct HamiltonianTerms{Hs <: Tuple{Vararg{Hamiltonian}}} <: Hamiltonian
    hs::Hs
end

"""
From the type initialize
"""
function HamiltonianTerms{HS}(g::IsingGraph) where HS <: Tuple{Vararg{Hamiltonian}}
    HamiltonianTerms{HS}(ntuple(i->HS.parameters[i](g), length(HS.parameters)))
end

function changeterm(hts, newham)
    hams = hamiltonians(hts)
    newhams = tuple((Base.typename(typeof(newham))  == Base.typename(typeof(h)) ? newham : h for h in hams)...)
    return HamiltonianTerms{typeof(newhams)}(newhams)
end

HamiltonianTerms(hs::Type{<:Hamiltonian}...) = HamiltonianTerms{Tuple{hs...}}
HamiltonianTerms(hs::Hamiltonian...) = HamiltonianTerms{Tuple{typeof.(hs)...}}(hs)

hamiltonians(hts::HamiltonianTerms) = getproperty(hts, :hamiltonians)

Base.:+(h1::Hamiltonian, h2::Hamiltonian) = HamiltonianTerms((h1, h2))
Base.:+(h1::Hamiltonian, h2::HamiltonianTerms) = HamiltonianTerms((h1, hamiltonians(h2)...))
Base.:+(h1::HamiltonianTerms, h2::Hamiltonian) = HamiltonianTerms((hamiltonians(h1)..., h2))

@generated function paramnames(h::HamiltonianTerms{Hs}) where Hs
    _paramnames = tuple(Iterators.Flatten(paramnames.(Hs.parameters)...)...)
    return :($_paramnames)
end

function setparam(ham::Hamiltonian, field, paramval)
    fnames = fieldnames(typeof(ham))
    found = findfirst(x->x==field, fnames)
    if isnothing(found)
        error("Field $field not found in Hamiltonian $ham")
    end
    newfields = (i == found ? paramval : ham.fieldnames[i] for i in eachindex(fnames))
    Base.typename(typeof(ham)).wrapper(newfields...)
end

function deactivateparam(ham::Hamiltonian, param::Symbol)
    initialparam = getproperty(ham, param)
    setparam(ham, param, deactivate(initialparam))
end

function deactivateparam(hts::HamiltonianTerms, param::Symbol)
    newham = deactivateparam(gethamiltonian(hts, param), param)
    changeterm(hts, newham)
end

"""
From the set of Hamiltonians, directly get a paramval from an underlying Hamiltonian
"""
# getparam(h::HamiltonianTerms, paramname::Symbol) = getparam(h, Val(paramname))
function Base.getproperty(h::HamiltonianTerms, paramname::Symbol)
    if paramname == :hamiltonians
        return getfield(h, :hs)
    end
    getparam(h, Val(paramname))
end

"""
Get a param
"""
@generated function getparam(h::HamiltonianTerms{Hs}, paramnameval::Val{paramname}) where {Hs, paramname}
    for (hidx, H) in enumerate(Hs.parameters)
        if paramname in fieldnames(H)
            return :(getfield(h,:hs)[$hidx].$(paramname))
        end
    end
    error("Parameter $paramname not found in any of the Hamiltonians")
end

"""
Get a hamiltonian from a type
"""
function gethamiltonian(hts::HamiltonianTerms, t::Type)
    for h in hamiltonians(hts)
        if typeof(h) <: t
            return h
        end
    end
    error("Type $t not found in any of the Hamiltonians")
end

"""
Get the hamiltonian from a parameter name
"""
function gethamiltonian(hts::HamiltonianTerms, t::Symbol)
    for h in hamiltonians(hts)
        if t in fieldnames(typeof(h))
            return h
        end
    end
    error("Type $t not found in any of the Hamiltonians")
end
export gethamiltonian

# Fallback for fieldnames for a hamltonian
Base.fieldnames(::Hamiltonian) = tuple()

"""
Iterating over terms forwards to the hamiltonians
"""
Base.iterate(hts::HamiltonianTerms, state = 1) = iterate(getfield(hts, :hs), state)

Base.broadcastable(c::HamiltonianTerms) = getfield(c, :hs)

"""
Get a hamiltonian from the set of hamiltonians
"""
Base.getindex(hts::HamiltonianTerms, idx) = getfield(hts, :hs)[idx]

"""
If updating functions are defined, update
"""
# @inline function update!(hts::HamiltonianTerms{Hs}, args) where Hs
#     @inline update!.(hamiltonians(hts), Ref(args))
# end
@inline function update!(hts::HamiltonianTerms{Hs}, args) where Hs
    hs = hamiltonians(hts)
    @inline _update!(gethead(hs),gettail(hs), args)
end

@inline function _update!(head, tail, args)
    @inline update!(head, args)
    @inline _update!(gethead(tail), gettail(tail), args)
end

@inline _update!(::Nothing, a, b) = nothing


@inline function init!(hts::HamiltonianTerms{Hs}, g) where Hs
    @inline init!.(hamiltonians(hts), Ref(g))
    return hts
end

init!(hts::Any, g) = hts


# Hamiltonian
function deltaH(hts::HamiltonianTerms)
    return reduce(+, deltaH.(hts))
end



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
include("GaussianBernoulli.jl")
include("Clamping.jl")
include("DepolarisationField.jl")




# CompositeHamiltonian(hamiltonians::Type{<:Hamiltonian}...) = CompositeHamiltonian{Tuple{hamiltonians...}}
# function Base.iterate(c::Type{<:Hamiltonian}, state = 1)
#     if state == 1
#         return c, 2
#     else
#         return nothing
#     end
# end
# Base.iterate(c::Type{<:CompositeHamiltonian}, state = 1) = iterate(c.parameters[1].parameters, state)
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

# params(::Type{T}, g::IsingGraph) where T <: Hamiltonian = params(T, eltype(g))
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
function addHparams!(graph, hamiltonian_params)
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
    graph.params = Parameters(;pairs...,get_nt(graph.params)...)
end
"""
Doesn't work if vector param is written as v_i
"""
function Hamiltonian_Builder(::Type{algo}, graph, hamiltonians::Hamiltonian) where {algo <: MCAlgorithm}
    addHparams!(graph, params(hamiltonians, graph))
    required_H = requires(algo)

    H_ex = H_expr(required_H, graph, hamiltonians...)
 
    H_ex = param_function(Meta.parse(H_ex), algo, graph.params)

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



### NEW SYSTEM

function dh(h::Hamiltonian, args; j)
    (;contractions, multiplications) = get_terms(h)
    return (args.newstate - args.gstate[j]) * contractions(args) + multiplications(args)
end