
abstract type LatticeType end
abstract type LayerTopology{U, Dim} end # U is periodicity type

include("Coordinates.jl")
include("LayerTopology.jl")
include("Distances.jl")