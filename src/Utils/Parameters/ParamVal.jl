export ParamVal, isinactive, isactive, toggle, default, getvalfield, setvalfield!, homogeneousval, description, toggle
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
abstract type AbstractParamVal{T, Default, Active, N} <: AbstractArray{T,N} end
mutable struct ParamVal{T, Default, Active, AT, N} <: AbstractParamVal{T, Default, Active, N}
    const val::AT
    size::NTuple{N, Int}
    description::String
end

# Special Cases:
# Vector like but same value everywhere
const HomogeneousParamVal{T, D, Active, N} = ParamVal{T, D, Active, <:AbstractArray{T,0} , N}
# Scalar Like/Reflike
const ScalarParamVal{T, D, Active} = ParamVal{T, D, Active, <:AbstractArray{T,0}, 0}
# Either, but inlined static value
const StaticParamVal{T, D, N} = ParamVal{T, D, false, <:AbstractArray{T,N}, N}


function ParamVal(val::T, default = nothing; size = nothing, active = false, description = "") where T
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
    return ParamVal{et, default, active, typeof(value), DIMS}(value, size, description)
end

# ParamVal{T, Default, Active, RD}(description::String = "") where {T, Default, Active, RD} = 
#     ParamVal{T, Default, Active, RD, (val isa AbstractArray ? length(size(val)) : 1)}(nothing, nothing, description)

ScalarParam(val::Real; description = "") = ScalarParam(typeof(val), val; description = description)
# ScalarParam(T::Type, val::Real; description = "") = ParamVal{T, T(val), true, Array{T,0}, 0}(fill(val), (), description)
ScalarParam(T::Type, val::Real; active = true, description = "") = ParamVal(fill(convert(T,val)), convert(T,val); active, size = tuple(), description = description)


"""
Stores a homogeneous value for vector like ParamVals
"""
function HomogeneousParam(val::Real, size...; active = true, description = "")
    @assert !isempty(size) "HomogeneousParam requires size arguments"
    return ParamVal(fill(val), val; size, active, description = description)
end

function StaticParam(val, size...; description = "")
    return ParamVal(zeros(typeof(val), size...), val, active = false, description = description)
end

# From other ParamVals
function ParamVal(p::ParamVal, default = nothing , active::Bool = nothing)
    isnothing(active) && (active = isactive(p))
    isnothing(default) && (default = default(p))
    return ParamVal(p.val, default; active, description = p.description)
end

"""
Paramval: Active -> Static
"""
toggle(p::ParamVal{T, Default, Active}) where {T, Default, Active} = ParamVal{T, Default, !Active}(p.val)

#Changing parameters
changeactivation(p::ParamVal{T}, activate) where T = ParamVal(p.val, default(p), active = activate, description = p.description)
activate(p::ParamVal{T}) where T = changeactivation(p, true)
deactivate(p::ParamVal{T}) where T = changeactivation(p, false)


## TRAITS
# isinactive(::ParamVal{A,B,C,N}) where {A,B,C,N}= !C
# isactive(::ParamVal{A,B,C,N}) where {A,B,C,N} = C
# isactive(::Type{ParamVal{A,B,C,N}}) where {A,B,C,N} = C
# isinactive(::Type{ParamVal{A,B,C,N}}) where {A,B,C,N} = !C


ishomogeneous(p::Type{<:ParamVal}) = p <: HomogeneousParamVal
ishomogeneous(p::ParamVal) = ishomogeneous(typeof(p))

isscalar(p::Union{Type{<:ParamVal}, ParamVal}) = dims(p) == 0
isstatic(p::Type{<:ParamVal{A,B,Active}}) where {A,B,Active} = !Active
isstatic(p::ParamVal) = isstatic(typeof(p))

isactive(p::ParamVal{A,B,C}) where {A,B,C} = C
isactive(p::Type{<:ParamVal{A,B,C}}) where {A,B,C} = C
isinactive(p::ParamVal{A,B,C}) where {A,B,C}= !C

@inline default(p::ParamVal{T, Default, Active, N}) where {T, Default, Active, N} = Default
@inline default(::Type{<:ParamVal{T, Default}}) where {T, Default} = Default
description(p::ParamVal) = p.description

dims(p::ParamVal{T, Default, Active, N}) where {T, Default, Active, N} = N
dims(::Type{ParamVal{T, Default, Active, N}}) where {T, Default, Active, N} = N

# Will be constant over any iteration
loopconstant(p::ParamVal) = !isactive(p) || ishomogeneous(p)
loopconstant(p::Type{<:ParamVal}) = !isactive(p) || ishomogeneous(p)
function unroll_exp(p::Union{Type{<:ParamVal}, <:ParamVal}, vecname, exp_f = identity)
    :(length(vecname)*$(exp_f(:($(vecname)[]))))
end

# # Single value Params
# Base.getindex(p::ParamVal) = p.val[]
# Base.getindex(p::ParamVal, idx) = p.val[idx]

function Base.setindex!(p::ParamVal, val)
    @assert !isstatic(p) "Cannot set value of a static ParamVal"
    @assert ishomogeneous(p) || isscalar(p) "Cannot set value of a non-homogeneous/scalar ParamVal without an index"
    p.val[] = val
end
function Base.setindex!(p::ParamVal, val, idx)
    @assert !isstatic(p) "Cannot set value of a static ParamVal"
    p.val[idx] = val
end

function Base.eachindex(p::ParamVal)
    if ishomogeneous(p)
        return Base.OneTo(prod(size(p)))
    end
    eachindex(p.val)
end

Base.size(p::ParamVal) = p.size

function Base.length(p::ParamVal)
    if ishomogeneous(p)
        return prod(size(p))
    end
    length(p.val)
end

Base.eltype(p::ParamVal) = eltype(p.val)
Base.eltype(pt::Type{<:ParamVal{T,D,A,N}}) where {T,D,A,N} = eltype(T)

"""
For getting and setting fields of the value of a ParamVal
"""
getvalfield(p::ParamVal, field) = getfield(p.val, field)
setvalfield!(p::ParamVal, field, val) = setfield!(p.val, field, val)


#Vector Like ParamVals
# @inline @generated function Base.getindex(p::ParamVal{T}) where T <: AbstractArray
#     @assert !isactive(p) || ishomogeneous(p) "Cannot index an active parameter with []"
#     getval = isstatic(p) ? :(p.val[]) : :(default(p))
#     return :($(getval)::eltype(T))
# end
@inline function Base.getindex(p::ParamVal{T}) where T <: AbstractArray
    @assert !isactive(p) || ishomogeneous(p) "Cannot index an active parameter with []"
    getval = isstatic(p) ? default(p) : p.val[]
    return getval::eltype(T)
end


@inline function Base.getindex(p::ParamVal{T}, idx::Integer) where T <: AbstractArray
    if ishomogeneous(p)
        @boundscheck 0 < idx <= prod(size(p))
        retval = p.val[]
    elseif isstatic(p)
        @boundscheck checkbounds(p.val, idx)
        retval = default(p)
    else
        retval = p.val[idx]
    end
    return retval::eltype(T)
end

@inline function Base.getindex(p::ParamVal{T}, idx::UnitRange) where T <: AbstractArray
    if ishomogeneous(p)
        @boundscheck 0 < first(idx) <= last(idx) <= prod(size(p))
        return fill(p.val[], size(p))::Vector{eltype(T)}
    elseif isstatic(p)
        @boundscheck checkbounds(p.val, idx)
        return fill(default(p), length(idx))::Vector{eltype(T)}
    else
        return p.val[idx]::Vector{eltype(T)}
    end
end

# @inline @generated function Base.setindex!(p::ParamVal{T}, val, idx) where T <: AbstractArray
#     if ishomogeneous(p)
#         return :((p.homogeneousval = val)::eltype(T))
#     end
#     return :((setindex!(p.val, val, idx))::T)
# end

@inline function Base.setindex!(p::ParamVal{T}, val, idx) where T <: AbstractArray
    @assert !isstatic(p) "Cannot set value of a static ParamVal, use StaticParamVal(param, val) instead"
    if ishomogeneous(p)
        @boundscheck 0 < idx <= prod(size(p))
        p.val[] = val
    else
        p.val[idx] = val
    end
end

@inline function Base.setindex!(p::ParamVal{T}, val) where T <: AbstractArray
    @assert !isstatic(p) "Cannot set value of a static ParamVal, use StaticParamVal(param, val) instead"
    @assert ishomogeneous(p) || isscalar(p) "Cannot set value of a non-homogeneous/scalar ParamVal without an index"
    p.val[] = val
end

Base.dotview(p::ParamVal{T}, i...) where T <: AbstractArray = Base.dotview(p.val, i...)
Base.materialize!(p::ParamVal{T}, a::Base.Broadcast.Broadcasted{<:Any}) where T <: AbstractArray = Base.materialize!(p.val, a)

@inline Base.lastindex(p::ParamVal{T}) where T <: AbstractArray = lastindex(p.val)
@inline Base.firstindex(p::ParamVal{T}) where T <: AbstractArray = firstindex(p.val)
@inline Base.eachindex(p::ParamVal{T}) where T <: AbstractArray = eachindex(p.val)
Base.length(p::ParamVal{T}) where T <: AbstractArray = length(p.val)
@inline Base.eltype(p::ParamVal{<:AbstractArray{T}}) where T = T
Base.splice!(p::ParamVal{T}, idx...) where T <: AbstractArray = splice!(p.val, idx...)

Base.push!(p::ParamVal{T}, val) where T <: AbstractArray = push!(p.val, val)

sethomogeneousval(p::ParamVal{T}, val) where T = HomogeneousParam(val, default(p), active = isactive(p), description = p.description)
removehomogeneousval(p::ParamVal{T}, def = default(p)) where T = ParamVal(fill(p[], size(p)...), def, isactive(p), description = p.description)


# Loopvectorization stuff
using LayoutPointers
LoopVectorization.check_args(p::ParamVal{T}) where T <: DenseArray = true
@inline Base.pointer(p::ParamVal{T}) where T <: DenseArray = pointer(p.val)
@inline LayoutPointers.memory_reference(p::ParamVal{T}) where T <: DenseArray = LayoutPointers.memory_reference(p.val)
@inline LayoutPointers.stridedpointer_preserve(p::ParamVal{T}) where T <: DenseArray = LayoutPointers.stridedpointer_preserve(p.val)
Base.strides(p::ParamVal{T}) where T <: DenseArray = strides(p.val)
# Base.IndexStyle(::Type{<:ParamVal}) = IndexLinear()
# Base.BroadcastStyle(::Type{ParamVal{T,A,B,C,D}}) where {T<:AbstractArray,A,B,C,D} = Broadcast.ArrayStyle{ParamVal}()
# Base.BroadcastStyle(::Type{ParamVal{T,A,B,C,D}}) where {T,A,B,C,D} = Broadcast.Style{ParamVal}()


vec_val_eltype(r::Real) = typeof(r)
vec_val_eltype(v::AbstractArray) = eltype(v)
vec_val_eltype(t::Type{<:Real}) = t
vec_val_eltype(t::Type{<:AbstractArray}) = eltype(t)
vec_val_eltype(v::ParamVal) = eltype(v)
vec_val_eltype(t::Type{<:ParamVal}) = eltype(t)
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
paramzero(::ParamVal{T}) where T = zero(T)
export paramzero

# Compact show for when ParamVal appears in other structs
function Base.show(io::IO, p::ParamVal{T}) where T
    if get(io, :compact, false)
        if ishomogeneous(p)
            print(io, "ParamVal($(p.val[]), len=$(length(p)))")
        else
            print(io, "ParamVal($(eltype(p))[...], len=$(length(p)))")
        end
    else
        print(io, "ParamVal{", T, "}(")
        print(io, p.description == "" ? "no description" : "\"$(p.description)\"")
        print(io, ", ", isactive(p) ? "active" : "inactive")
        if ishomogeneous(p)
            print(io, ", val=$(p.val[]), len=$(length(p)))")
        else
            print(io, ", len=$(length(p)))")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::ParamVal{T}) where T
    print(io, (isactive(p) ? "Active " : "Inactive "))
    ishomogeneous(p) && print(io, "Homogeneous ")
    println(io, "$(p.description) with value: ")
    if isscalar(p)
        print(io, "$(p.val[])")
    else
        print(io, p[1:end])
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::ParamVal{T}) where {T <: AbstractVector}
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
