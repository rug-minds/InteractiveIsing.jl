export ParamVal, isinactive, isactive, toggle, default

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

isinactive(p::ParamVal{A,B,C}) where {A,B,C}= !C
isactive(p::ParamVal{A,B,C}) where {A,B,C} = C
toggle(p::ParamVal{T, Default, Active}) where {T, Default, Active} = ParamVal{T, Default, !Active}(p.val)
@inline default(p::ParamVal{T, Default}) where {T, Default} = Default
@inline Base.getindex(p::ParamVal) = p.val
@inline Base.setindex!(p::ParamVal, val) = (p.val = val)
@inline Base.getindex(p::ParamVal{T}, idx) where T <: Vector = p.val[idx]
@inline Base.setindex!(p::ParamVal{T}, val, idx) where T <: Vector = (p.val[idx] = val)

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


export ParamVal, toggle