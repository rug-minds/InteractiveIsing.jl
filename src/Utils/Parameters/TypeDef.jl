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
struct ParamTensor{T, Default, Active, AT, N} <: AbstractParamTensor{T, Default, Active, N}
    val::AT
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

