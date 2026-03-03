#=
@turbo compatible homogenous arrays
=#
using LayoutPointers
using LayoutPointers: StaticInt
import VectorizationBase

using VectorizationBase: AbstractSIMD, VecUnroll, vload
const _SAI = LayoutPointers.StaticArrayInterface

include("StaticFill.jl")
include("FillArray.jl")