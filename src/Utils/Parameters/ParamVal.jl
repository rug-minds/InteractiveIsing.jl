export ParamVal, isinactive, isactive, toggle, default, getvalfield, setvalfield!, runtimeglobal, description
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
mutable struct ParamVal{T, Default, Active, RD, N} <: AbstractArray{T, N}
    val::T
    runtimeglobal::RD # For vector valued parameters, a global value that can be changed at runtime
    description::String
end

function ParamVal(val::T, default, description = "", active = false; runtimeglobal = false) where T
    # If val is vector type, default value must be eltype, 
    # otherwise it must be the same type
    DIMS = nothing
    if val isa AbstractArray
       DIMS = length(size(val))
    else
        DIMS = 1
    end    
    
    if T <: Vector 
        et = eltype(T)
        default = convert(eltype(T), default)
        if runtimeglobal
            return ParamVal{T, default, active, Base.RefValue{et}, DIMS}(val, Ref(default), description)
        else
            return ParamVal{T, default, active, Nothing, DIMS}(val, nothing, description)
        end
    else
        default = convert(T, default)
        value = Ref(val)
        return ParamVal{typeof(value), default, active, Nothing, DIMS}(value, nothing, description)
    end
end

function GlobalParamVal(val, length = 0, description = "", active = false)
    return ParamVal(zeros(eltype(val), length), zero(eltype(val)), description, active; runtimeglobal = true)
end

function DefaultParamVal(val, description = "")
    return ParamVal(typeof(val)[], val, description, false)
end


ParamVal{T, Default, Active, RD}(description::String = "") where {T, Default, Active, RD} = ParamVal{T, Default, Active, RD, (val isa AbstractArray ? length(size(val)) : 1)}(nothing, nothing, description)
function ParamVal{T, Default, Active, RD, N}(description::String = "") where {T, Default, Active, RD, N}
    val = nothing
    rtglobal = nothing
    if T <: AbstractArray
        val = T[]
    elseif T <: Number
        val = T(0)
    end
    if !(RD <: Nothing)
        rtglobal = Ref(Default)
    end
    ParamVal{T, Default, Active, RD, N}(val, nothing, description)
end


function ParamVal(p::ParamVal, active::Bool = nothing)
    return ParamVal(p.val, default(p), p.description, precedence_val(active, isactive(p)))
end

function ParamVal(p::ParamVal, default, active::Bool = nothing)
    isnothing(active) && (active = isactive(p))
    return ParamVal(p.val, default, p.description, active)
end


## TRAITS
isinactive(::ParamVal{A,B,C}) where {A,B,C}= !C
isactive(::ParamVal{A,B,C}) where {A,B,C} = C
isactive(::Type{ParamVal{A,B,C,D,N}}) where {A,B,C,D,N} = C
isinactive(::Type{ParamVal{A,B,C,D,N}}) where {A,B,C,D,N} = !C
@inline default(p::ParamVal{T, Default, Active, D, N}) where {T, Default, Active, D, N} = Default
@inline default(::Type{ParamVal{T, Default, Active, D, N}}) where {T, Default, Active, D, N} = Default
description(p::ParamVal) = p.description
runtimeglobal(p::ParamVal{T, Default, Active, RD}) where {T, Default, Active, RD} = RD != Nothing
runtimeglobal(::Type{ParamVal{T, Default, Active, RD, N}}) where {T, Default, Active, RD, N} = RD != Nothing

function get_globalval(p::ParamVal{T, Default, Active, RD, N}) where {T, Default, Active, RD, N}
    rd = getfield(p, :runtimeglobal)::RD
    return rd[]::eltype(T)
end


dims(p::ParamVal{T, Default, Active, RD, N}) where {T, Default, Active, RD, N} = N
dims(::Type{ParamVal{T, Default, Active, RD, N}}) where {T, Default, Active, RD, N} = N

# Will be constant over any iteration
loopconstant(p::ParamVal) = !isactive(p) || runtimeglobal(p)
loopconstant(p::Type{<:ParamVal}) = !isactive(p) || runtimeglobal(p)
function unroll_exp(p::Union{Type{<:ParamVal}, <:ParamVal}, vecname, exp_f = identity)
    :(length(vecname)*$(exp_f(:($(vecname)[]))))
end


toggle(p::ParamVal{T, Default, Active}) where {T, Default, Active} = ParamVal{T, Default, !Active}(p.val)

# Single value Params
@inline Base.getindex(p::ParamVal{T}) where T <: Ref =p.val[]
@inline Base.setindex!(p::ParamVal{T}, val) where T <: Ref = (p.val[] = val)
@inline Base.lastindex(p::ParamVal{T}) where T <: Ref = 1
@inline Base.eachindex(p::ParamVal{T}) where T <: Ref = Base.OneTo(1)
Base.size(p::ParamVal{T}) where T = (1,)
Base.length(p::ParamVal{T}) where T = 1
@inline Base.eltype(p::ParamVal{T}) where T = T
@inline Base.eltype(pt::Type{<:ParamVal{T,D,A,RD,N}}) where {T,D,A,RD,N} = eltype(T)

"""
For getting and setting fields of the value of a ParamVal
"""
getvalfield(p::ParamVal, field) = getfield(p.val, field)
setvalfield!(p::ParamVal, field, val) = setfield!(p.val, field, val)


#Vector Like ParamVals
function Base.getindex(p::ParamVal{T}) where T <: AbstractArray
    if isactive(p) && !runtimeglobal(p)
        error("Cannot index an active parameter with []")
    end
    if runtimeglobal(p)
        return p.runtimeglobal[]::eltype(T)
    else
        return default(p)::eltype(T)
    end
end


@inline function Base.getindex(p::ParamVal{T}, idx) where T <: AbstractArray
    if isactive(p)
        if runtimeglobal(p)
            return p.runtimeglobal[]::eltype(T)
        else
            return getindex(p.val, idx)::eltype(T)
        end
    else
        return default(p)::eltype(T)
    end
end

@inline function Base.getindex(p::ParamVal{T}, idx::UnitRange) where T <: AbstractArray
    if isactive(p)
        if runtimeglobal(p)
            return [p.runtimeglobal[] for i in idx]::Vector{eltype(T)}
        else
            return getindex(p.val, idx)::Vector{eltype(T)}
        end
    else
        return [default(p) for i in idx]::Vector{eltype(T)}
    end
end

@inline function Base.setindex!(p::ParamVal{T}, val, idx...) where T <: AbstractArray
    if runtimeglobal(p)
        return p.runtimeglobal[] = val
    end
    return setindex!(p.val, val, idx...)::eltype(T)
end

Base.dotview(p::ParamVal{T}, i...) where T <: AbstractArray = Base.dotview(p.val, i...)
Base.materialize!(p::ParamVal{T}, a::Base.Broadcast.Broadcasted{<:Any}) where T <: AbstractArray = Base.materialize!(p.val, a)

@inline Base.lastindex(p::ParamVal{T}) where T <: AbstractArray = lastindex(p.val)
@inline Base.firstindex(p::ParamVal{T}) where T <: AbstractArray = firstindex(p.val)
@inline Base.eachindex(p::ParamVal{T}) where T <: AbstractArray = eachindex(p.val)
Base.size(p::ParamVal{T}) where T <: AbstractArray = size(p.val)
Base.length(p::ParamVal{T}) where T <: AbstractArray = length(p.val)
@inline Base.eltype(p::ParamVal{<:AbstractArray{T}}) where T = T
Base.splice!(p::ParamVal{T}, idx...) where T <: AbstractArray = splice!(p.val, idx...)

Base.push!(p::ParamVal{T}, val) where T <: AbstractArray = push!(p.val, val)


#Changing parameters
changeactivation(p::ParamVal{T}, activate) where T = ParamVal(p.val, default(p), p.description, activate)
activate(p::ParamVal{T}) where T = changeactivation(p, true)
deactivate(p::ParamVal{T}) where T = changeactivation(p, false)

setruntimeglobal(p::ParamVal{T}, val) where T = Paramval(p.val, default(p), p.description, isactive(p), runtimeglobal = val)
removeruntimeglobal(p::ParamVal{T}) where T = Paramval(p.val, default(p), p.description, isactive(p), runtimeglobal = false)


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


function Base.show(io::IO, ::MIME"text/plain", p::ParamVal{T}) where T
    print(io, (isactive(p) ? "Active " : "Inactive "))
    print(io, "$(p.description) with value: ")
    println(io, "$(p.val)")
    print(io, "Defaulting to: $(default(p))")
end

function Base.show(io::IO, ::MIME"text/plain", p::ParamVal{T}) where {T <: AbstractVector}
    if runtimeglobal(p)
        l = length(p.val)
        println(io, "$(l)-element $(eltype(p.val)) constant parameter")
        print(io, "Value: $(p.runtimeglobal[])")
    else
        println(io, (isactive(p) ? "Active " : "Inactive "))
        println(io, "$(p.description) with vector value.")
        println(io, "Defaulting to: $(default(p))")
        display(p.val)
    end
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