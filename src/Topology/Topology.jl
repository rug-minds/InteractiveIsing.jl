
abstract type LatticeType end
abstract type AbstractLayerTopology{U, Dim} end # U is periodicity type

include("Defs.jl")
include("Interface.jl")
include("Coordinates.jl")
include("SquareTopology.jl")
include("LayerTopology.jl")
include("Distances.jl")