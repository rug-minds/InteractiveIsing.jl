abstract type AbstractProposer end
abstract type AbstractProposal end

include("IsingProposer.jl")
include("LocalProposer.jl")
include("FlipProposal.jl")
include("VectorSpinProposer.jl")
