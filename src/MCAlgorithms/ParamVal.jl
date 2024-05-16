export ParamVal, isinactive, isactive, toggle, default
"""
A value for the parameters of a Hamiltonian
It holds a description and a value of type t
It also stores wether it's active and if not a fallback value
    so that functions can be compiled with the default value inlined
    to save runtime when the parameter is inactive
    E.g. a parameter might be a vector, but if it's inactive
    the whole vector can be set to a constant value, so that
    memory does not need to be accessed.
"""
mutable struct ParamVal{T, Default, Active}
    val::T
    description::String
end

function ParamVal(val::T, default, description, active = false) where T
    # If val is vector type, default value must be eltype, 
    # otherwise it must be the same type
    if T <: Vector 
        default = convert(eltype(T), default)
    else
        default = convert(T, default)
    end
    
    return ParamVal{T, default, active}(val, description)
end

function ParamVal(p::ParamVal, active = nothing)
    return ParamVal(p.val, default(p), p.description, precedence_val(active, isactive(p)))
end


isinactive(::ParamVal{A,B,C}) where {A,B,C}= !C
isactive(::ParamVal{A,B,C}) where {A,B,C} = C
toggle(p::ParamVal{T, Default, Active}) where {T, Default, Active} = ParamVal{T, Default, !Active}(p.val)
@inline default(::ParamVal{T, Default}) where {T, Default} = Default
@inline Base.eltype(::ParamVal{T}) where T = T
@inline Base.getindex(p::ParamVal) = p.val
@inline Base.setindex!(p::ParamVal, val) = (p.val = val)
@inline Base.getindex(p::ParamVal{T}, idx) where T <: Real = p.val
@inline Base.setindex!(p::ParamVal{T}, val, idx) where T <: Real = (p.val = val)
@inline Base.getindex(p::ParamVal{T}, idx) where T <: Vector = p.val[idx]
@inline Base.setindex!(p::ParamVal{T}, val, idx) where T <: Vector = (p.val[idx] = val)

@inline Base.size(::ParamVal{T}) where T <: Real = (1,)
@inline Base.size(p::ParamVal{T}) where T <: Vector = size(p.val)

# @inline function setparam(p::NamedTuple, symbol, val, active = nothing)
#     @assert haskey(p, symbol)
#     # newval = 
#     # return (;p...， symbol => ParamVal(p[symbol], val, active))
# end


function Base.show(io::IO, p::ParamVal{T}) where T
    print(io, (isactive(p) ? "Active " : "Inactive "))
    print(io, "$(p.description) with value: ")
    println(io, "$(p.val)")
    print(io, "Defaulting to: $(default(p))")
end

function Base.show(io::IO, p::ParamVal{T}) where {T <: Vector}
    print(io, (isactive(p) ? "Active " : "Inactive "))
    println(io, "$(p.description) with vector value.")
    println(io, "Defaulting to: $(default(p))")
    display(p.val)
end

"""
Adds the paramvals to g.params, overwrites the old ones
"""
function addparams!(graph, hamiltonian_params)
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
    graph.params = (;graph.params..., pairs...)
end

"""
If one of the values is nothing, return the other, otherwise return the logical and of the two values
"""
function nothing_and(val1, val2)
    if isnothing(val1)
        return val2
    elseif isnothing(val2)
        return val1
    else
        return val1 && val2
    end
end

"""
If one of the values is nothing, return the other, otherwise return the logical or of the two values
"""
function nothing_or(val1, val2)
    if isnothing(val1)
        return val2
    elseif isnothing(val2)
        return val1
    else
        return val1 || val2
    end
end

"""
If the first value is nothing, return the second value, otherwise return the first value
"""
function precedence_val(val1, val2)
    if isnothing(val1)
        return val2
    else
        return val1
    end
end


export ParamVal, toggle