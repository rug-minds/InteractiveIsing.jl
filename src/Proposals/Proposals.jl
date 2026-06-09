abstract type AbstractProposer end
abstract type AbstractProposal end

include("IsingProposer.jl")
include("LocalProposer.jl")
include("SingleSpinProposal.jl")
include("MultiSpinProposal.jl")
include("ProposedState.jl")
