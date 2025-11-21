export ParamTensor, isinactive, isactive, toggle, default, getvalfield, setvalfield!, homogeneousval, description, toggle
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
abstract type AbstractParamTensor{T, Default, Active, N} <: AbstractArray{T,N} end
mutable struct ParamTensor{T, Default, Active, AT, N} <: AbstractParamTensor{T, Default, Active, N}
    const val::AT
    size::NTuple{N, Int}
    description::String
end

# Special Cases:
# Vector like but same value everywhere
const HomogeneousParamTensor{T, D, Active, N} = ParamTensor{T, D, Active, <:AbstractArray{T,0} , N}
# Scalar Like/Reflike
const ScalarParamTensor{T, D, Active} = ParamTensor{T, D, Active, <:AbstractArray{T,0}, 0}
# Either, but inlined static value
const StaticParamTensor{T, D, N} = ParamTensor{T, D, false, <:AbstractArray{T,N}, N}


function ParamTensor(val::T, default = nothing; size = nothing, active = false, description = "") where T
    # # If val is vector type, default value must be eltype, 
    # # otherwise it must be the same type
    # DIMS = nothing
    # if val isa AbstractArray
    #    DIMS = length(Base.size(val))
    # else
    #     DIMS = 0
    # end    
    
    value = val
    if T <: Array #
        et = eltype(T)
        default = default == nothing ? et(1) : convert(eltype(T), default)
    else
        default = default == nothing ? T(1) : convert(T, default)
        value = fill(val)
        et = T
    end

    isnothing(size) && (size = Base.size(value))
    DIMS = length(size)
    return ParamTensor{et, default, active, typeof(value), DIMS}(value, size, description)
end

# ParamTensor{T, Default, Active, RD}(description::String = "") where {T, Default, Active, RD} = 
#     ParamTensor{T, Default, Active, RD, (val isa AbstractArray ? length(size(val)) : 1)}(nothing, nothing, description)

ScalarParam(val::Real; description = "") = ScalarParam(typeof(val), val; description = description)
# ScalarParam(T::Type, val::Real; description = "") = ParamTensor{T, T(val), true, Array{T,0}, 0}(fill(val), (), description)
ScalarParam(T::Type, val::Real; active = true, description = "") = ParamTensor(fill(convert(T,val)), convert(T,val); active, size = tuple(), description = description)


"""
Stores a homogeneous value for vector like ParamTensors
"""
function HomogeneousParam(val::Real, size::Integer...; default = val, active = true, description = "")
    @assert !isempty(size) "HomogeneousParam requires size arguments"
    return ParamTensor(fill(val), default; size, active, description = description)
end

function StaticParam(val, size...; description = "")
    return ParamTensor(zeros(typeof(val), size...), val, active = false, description = description)
end

# From other ParamTensors
function ParamTensor(p::ParamTensor, default = nothing , active::Bool = nothing)
    isnothing(active) && (active = isactive(p))
    isnothing(default) && (default = default(p))
    return ParamTensor(p.val, default; active, description = p.description)
end

"""
ParamTensor: Active -> Static
"""
toggle(p::ParamTensor{T, Default, Active}) where {T, Default, Active} = ParamTensor{T, Default, !Active}(p.val)

#Changing parameters
changeactivation(p::ParamTensor{T}, activate) where T = ParamTensor(p.val, default(p), active = activate, description = p.description)
activate(p::ParamTensor{T}) where T = changeactivation(p, true)
deactivate(p::ParamTensor{T}) where T = changeactivation(p, false)


## TRAITS
# isinactive(::ParamTensor{A,B,C,N}) where {A,B,C,N}= !C
# isactive(::ParamTensor{A,B,C,N}) where {A,B,C,N} = C
# isactive(::Type{ParamTensor{A,B,C,N}}) where {A,B,C,N} = C
# isinactive(::Type{ParamTensor{A,B,C,N}}) where {A,B,C,N} = !C


ishomogeneous(p::Type{<:ParamTensor}) = p <: HomogeneousParamTensor
ishomogeneous(p::ParamTensor) = ishomogeneous(typeof(p))

isscalar(p::Union{Type{<:ParamTensor}, ParamTensor}) = dims(p) == 0
isstatic(p::Type{<:ParamTensor{A,B,Active}}) where {A,B,Active} = !Active
isstatic(p::ParamTensor) = isstatic(typeof(p))

isactive(p::ParamTensor{A,B,C}) where {A,B,C} = C
isactive(p::Type{<:ParamTensor{A,B,C}}) where {A,B,C} = C
isinactive(p::ParamTensor{A,B,C}) where {A,B,C}= !C

@inline default(p::ParamTensor{T, Default, Active, N}) where {T, Default, Active, N} = Default
@inline default(::Type{<:ParamTensor{T, Default}}) where {T, Default} = Default
description(p::ParamTensor) = p.description

dims(p::ParamTensor{T, Default, Active, N}) where {T, Default, Active, N} = N
dims(::Type{ParamTensor{T, Default, Active, N}}) where {T, Default, Active, N} = N

# Will be constant over any iteration
loopconstant(p::ParamTensor) = !isactive(p) || ishomogeneous(p)
loopconstant(p::Type{<:ParamTensor}) = !isactive(p) || ishomogeneous(p)
function unroll_exp(p::Union{Type{<:ParamTensor}, <:ParamTensor}, vecname, exp_f = identity)
    :(length(vecname)*$(exp_f(:($(vecname)[]))))
end

# # Single value Params
# Base.getindex(p::ParamTensor) = p.val[]
# Base.getindex(p::ParamTensor, idx) = p.val[idx]


function Base.eachindex(p::ParamTensor)
    if ishomogeneous(p)
        return Base.OneTo(prod(size(p)))
    end
    eachindex(p.val)
end

Base.size(p::ParamTensor) = p.size

function Base.length(p::ParamTensor)
    if ishomogeneous(p)
        return prod(size(p))
    end
    length(p.val)
end

Base.eltype(pt::Type{<:ParamTensor{T,D,A,N}}) where {T,D,A,N} = eltype(T)

"""
For getting and setting fields of the value of a ParamTensor
"""
getvalfield(p::ParamTensor, field) = getfield(p.val, field)
setvalfield!(p::ParamTensor, field, val) = setfield!(p.val, field, val)


#Vector Like ParamTensors
# @inline @generated function Base.getindex(p::ParamTensor{T}) where T <: AbstractArray
#     @assert !isactive(p) || ishomogeneous(p) "Cannot index an active parameter with []"
#     getval = isstatic(p) ? :(p.val[]) : :(default(p))
#     return :($(getval)::eltype(T))
# end
@inline function Base.getindex(p::ParamTensor{T}) where T
    @assert !isactive(p) || ishomogeneous(p) "Cannot index an active parameter with []"
    getval = isstatic(p) ? default(p) : p.val[]
    return getval::T
end


@inline function Base.getindex(p::ParamTensor{T}, idx::Integer) where T
    if ishomogeneous(p)
        @boundscheck 0 < idx <= prod(size(p))
        retval = p.val[]
    elseif isstatic(p)
        @boundscheck checkbounds(p.val, idx)
        retval = default(p)
    else
        retval = p.val[idx]
    end
    return retval::T
end

@inline function Base.getindex(p::ParamTensor{T}, idx::UnitRange) where T
    if ishomogeneous(p)
        @boundscheck 0 < first(idx) <= last(idx) <= prod(size(p))
        return fill(p.val[], size(p))::Vector{T}
    elseif isstatic(p)
        @boundscheck checkbounds(p.val, idx)
        return fill(default(p), length(idx))::Vector{T}
    else
        return p.val[idx]::Vector{T}
    end
end


@inline function Base.setindex!(p::ParamTensor{T}, val, idx) where T
    @assert !isstatic(p) "Cannot set value of a static ParamTensor, use StaticParamTensor(param, val) instead"
    if ishomogeneous(p)
        @boundscheck 0 < idx <= prod(size(p))
        p.val[] = val
    else
        p.val[idx] = val
    end
end

@inline function Base.setindex!(p::ParamTensor{T}, val) where T
    @assert !isstatic(p) "Cannot set value of a static ParamTensor, use StaticParamTensor(param, val) instead"
    @assert ishomogeneous(p) || isscalar(p) "Cannot set value of a non-homogeneous/scalar ParamTensor without an index"
    p.val[] = val
end

Base.dotview(p::ParamTensor{T}, i...) where T = Base.dotview(p.val, i...)
Base.materialize!(p::ParamTensor{T}, a::Base.Broadcast.Broadcasted{<:Any}) where T = Base.materialize!(p.val, a)

@inline Base.lastindex(p::ParamTensor{T}) where T = lastindex(p.val)
@inline Base.firstindex(p::ParamTensor{T}) where T = firstindex(p.val)
# @inline Base.eachindex(p::ParamTensor{T}) where T = eachindex(p.val)
@inline Base.eltype(p::ParamTensor{T}) where T = T
Base.splice!(p::ParamTensor{T}, idx...) where T = splice!(p.val, idx...)
Base.push!(p::ParamTensor{T}, val) where T = push!(p.val, val)

function sethomogeneoustensor(p::ParamTensor{T}, val) where T
    val = convert(T, val)
    size = Base.size(p)
    HomogeneousParam(val, size..., default = default(p), active = true, description = p.description)
end
function removehomogeneousval(p::ParamTensor{T}, def = default(p)) where T
    def = convert(T, def)
    ParamTensor(fill(p[], size(p)...), def; active = true, description = p.description)
end


# Loopvectorization stuff
using LayoutPointers
LoopVectorization.check_args(p::ParamTensor{T,A,B,V}) where {T,A,B,V <: DenseArray} = true
@inline Base.pointer(p::ParamTensor{T,A,B,V}) where {T,A,B,V <: DenseArray} = pointer(p.val)
@inline LayoutPointers.memory_reference(p::ParamTensor{T,A,B,V}) where {T,A,B,V <: DenseArray} = LayoutPointers.memory_reference(p.val)
@inline LayoutPointers.stridedpointer_preserve(p::ParamTensor{T,A,B,V}) where {T,A,B,V <: DenseArray} = LayoutPointers.stridedpointer_preserve(p.val)
Base.strides(p::ParamTensor{T,A,B,V}) where {T,A,B,V <: DenseArray} = strides(p.val)
# Base.IndexStyle(::Type{<:ParamTensor}) = IndexLinear()
# Base.BroadcastStyle(::Type{ParamTensor{T,A,B,C,D}}) where {T<:AbstractArray,A,B,C,D} = Broadcast.ArrayStyle{ParamTensor}()
# Base.BroadcastStyle(::Type{ParamTensor{T,A,B,C,D}}) where {T,A,B,C,D} = Broadcast.Style{ParamTensor}()


vec_val_eltype(r::Real) = typeof(r)
vec_val_eltype(v::AbstractArray) = eltype(v)
vec_val_eltype(t::Type{<:Real}) = t
vec_val_eltype(t::Type{<:AbstractArray}) = eltype(t)
vec_val_eltype(v::ParamTensor) = eltype(v)
vec_val_eltype(t::Type{<:ParamTensor}) = eltype(t)
"""
For vector like objects, find the promote type of the eltypes
"""
@generated function promote_eltype(vector_types...)
    t = promote_type(vec_val_eltype.(vector_types)...)
    return :($t)
end




"""
Gives the zero value of the type of the parameter
Or just zero of the number type if it's a number
This works with inlining of default values.
"""
paramzero(val::Any) = typeof(val)(0)
paramzero(::ParamTensor{T}) where T = zero(T)
export paramzero

# Compact show for when ParamTensor appears in other structs
function Base.show(io::IO, p::ParamTensor{T}) where T
    if get(io, :compact, false)
        if ishomogeneous(p)
            print(io, "ParamTensor($(p.val[]), len=$(length(p)))")
        else
            print(io, "ParamTensor($(eltype(p))[...], len=$(length(p)))")
        end
    else
        print(io, "ParamTensor{", T, "}(")
        print(io, p.description == "" ? "no description" : "\"$(p.description)\"")
        print(io, ", ", isactive(p) ? "active" : "inactive")
        if ishomogeneous(p)
            print(io, ", val=$(p.val[]), len=$(length(p)))")
        else
            print(io, ", len=$(length(p)))")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::ParamTensor{T}) where T
    print(io, (isactive(p) ? "Active " : "Inactive "))
    ishomogeneous(p) && print(io, "Homogeneous ")
    println(io, "$(p.description) with value: ")
    if isscalar(p)
        print(io, "$(p.val[])")
    else
        print(io, p[1:end])
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::ParamTensor{T}) where {T <: AbstractVector}
    if ishomogeneous(p)
        l = length(p.val)
        println(io, "$(l)-element $(eltype(p.val)) constant parameter")
        print(io, "Value: $(p.homogeneousval[])")
    else
        println(io, (isactive(p) ? "Active " : "Inactive "))
        println(io, "$(p.description) with vector value.")
        println(io, "Defaulting to: $(default(p))")
        display(p.val)
    end
end
