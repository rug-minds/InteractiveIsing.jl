#=
@turbo compatible homogenous arrays
=#
export ConstFill, ConstVal, UniformArray, OffsetArray, filltype

filltype(::Type{Vector}, val, size...) = fill(val, size...)


include("ConstFill.jl")
include("UniformArray.jl")
include("OffsetArray.jl")
