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
struct ParamVal{T, Default, Active, RD} <: DenseArray{T, 1}
    val::T
    runtimeglobal::RD
    description::String
end

function ParamVal(val::T, default, description = "", active = false; runtimeglobal = false) where T
    # If val is vector type, default value must be eltype, 
    # otherwise it must be the same type
    if T <: Vector 
        et = eltype(T)
        default = convert(eltype(T), default)
        if runtimeglobal
            return ParamVal{T, default, active, Ref{et}}(val, Ref(default), description)
        else
            return ParamVal{T, default, active, Nothing}(val, nothing, description)
        end
    else
        default = convert(T, default)
        return ParamVal{T, default, active, Nothing}(Ref(val), nothing, description, )
    end
    
    
end

function ParamVal(p::ParamVal, active::Bool = nothing)
    return ParamVal(p.val, default(p), p.description, precedence_val(active, isactive(p)))
end

function ParamVal(p::ParamVal, default, active::Bool = nothing)
    isnothing(active) && (active = isactive(p))
    return ParamVal(p.val, default, p.description, active)
end


isinactive(::ParamVal{A,B,C}) where {A,B,C}= !C
isactive(::ParamVal{A,B,C}) where {A,B,C} = C
isactive(::Type{ParamVal{A,B,C,D}}) where {A,B,C,D} = C
isinactive(::Type{ParamVal{A,B,C,D}}) where {A,B,C,D} = !C
default(::Type{ParamVal{T, Default, Active, D}}) where {T, Default, Active, D} = Default
description(p::ParamVal) = p.description
runtimeglobal(p::ParamVal{T, Default, Active, RD}) where {T, Default, Active, RD} = RD != Nothing
runtimeglobal(::Type{ParamVal{T, Default, Active, RD}}) where {T, Default, Active, RD} = RD != Nothing

toggle(p::ParamVal{T, Default, Active}) where {T, Default, Active} = ParamVal{T, Default, !Active}(p.val)
@inline default(::ParamVal{T, Default}) where {T, Default} = Default
# @inline function Base.getindex(p::ParamVal, i = nothing)
#     @assert isnothing(i) || i == 1
#     return p.val
# end

#General

#Values
# @inline Base.setindex!(p::ParamVal, val) = (p.val = val)
# @inline function Base.getindex(p::ParamVal{T}) where T
#     if isactive(p)
#         return p.val
#     else
#         return default(p)
#     end
# end
# @inline Base.setindex!(p::ParamVal{T}, val, idx) where T <: Real = (p.val = val)
# @inline Base.eachindex(p::ParamVal{T}) where T = Base.OneTo(1)
#Ref
@inline Base.getindex(p::ParamVal{T}) where T <: Ref =p.val[]
@inline Base.setindex!(p::ParamVal{T}, val) where T <: Ref = (p.val[] = val)
@inline Base.lastindex(p::ParamVal{T}) where T <: Ref = 1
@inline Base.eachindex(p::ParamVal{T}) where T <: Ref = Base.OneTo(1)
Base.size(p::ParamVal{T}) where T = (1,)
Base.length(p::ParamVal{T}) where T = 1
@inline Base.eltype(p::ParamVal{T}) where T = T



#Vectors
@inline function Base.getindex(p::ParamVal{T}, idx) where T <: AbstractVector
    if isactive(p)
        getindex(p.val, idx)::eltype(T)
    else
        if runtimeglobal(p)
            return p.runtimeglobal[]
        else
            return default(p)
        end
    end
end
@inline function Base.setindex!(p::ParamVal{T}, val, idx) where T <: AbstractVector
    if idx isa Real
        if !isactive(p) && runtimeglobal(p)
            return p.runtimeglobal[] = val
        end
    end
    return p.val[idx] = val
end
@inline Base.lastindex(p::ParamVal{T}) where T <: AbstractVector = lastindex(p.val)
@inline Base.firstindex(p::ParamVal{T}) where T <: AbstractVector = firstindex(p.val)
@inline Base.eachindex(p::ParamVal{T}) where T <: AbstractVector = eachindex(p.val)
Base.size(p::ParamVal{T}) where T <: AbstractVector = size(p.val)
Base.length(p::ParamVal{T}) where T <: AbstractVector = length(p.val)
@inline Base.eltype(p::ParamVal{<:AbstractVector{T}}) where T = T

Base.push!(p::ParamVal{T}, val) where T <: AbstractVector = push!(p.val, val)
LoopVectorization.check_args(p::ParamVal{T}) where T <: AbstractVector = true
@inline Base.pointer(p::ParamVal{T}) where T = pointer(p.val)
"""
For vector like objects, find the promote type of the eltypes
"""
promote_eltype(vector_types...) = promote_type(eltype.(vector_types)...)








"""
Gives the zero value of the type of the parameter
Or just zero of the number type if it's a number
This works with inlining of default values.
"""
paramzero(val::Any) = typeof(val)(0)
paramzero(::ParamVal{T}) where T = zero(T)
export paramzero


Base.BroadcastStyle(::Type{ParamVal{T,A,B}}) where {T<:AbstractArray,A,B} = Broadcast.ArrayStyle{ParamVal}()
Base.BroadcastStyle(::Type{ParamVal{T,A,B}}) where {T,A,B} = Broadcast.Style{ParamVal}()


function Base.show(io::IO, p::ParamVal{T}) where T
    print(io, (isactive(p) ? "Active " : "Inactive "))
    print(io, "$(p.description) with value: ")
    println(io, "$(p.val)")
    print(io, "Defaulting to: $(default(p))")
end

function Base.show(io::IO, p::ParamVal{T}) where {T <: AbstractVector}
    print(io, (isactive(p) ? "Active " : "Inactive "))
    println(io, "$(p.description) with vector value.")
    println(io, "Defaulting to: $(default(p))")
    display(p.val)
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