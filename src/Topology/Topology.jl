
abstract type LatticeType end
abstract type AbstractLayerTopology{U, Dim} end # U is periodicity type

include("Defs.jl")
include("Interface.jl")
include("Coordinates.jl")
include("Square/SquareTopology.jl")
include("Hexagonal/HexagonalTopology.jl")
include("LayerTopology.jl")
include("WoorldCoordinates.jl")
include("Distances.jl")
