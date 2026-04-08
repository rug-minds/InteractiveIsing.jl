module IsingLearning

using LuxCore
using Random: AbstractRNG
using SparseArrays

using DataStructures
using ChainRules, ChainRulesCore

import LuxCore: initialparameters, initialstates

# include local interactivesising
include(joinpath(@__DIR__, "..", "..","..", "src", "InteractiveIsing.jl"))
import .InteractiveIsing.Processes: init, step!
using .InteractiveIsing
using .InteractiveIsing: state, adj, setparam!, getparam, setSpins!, nStates
using .InteractiveIsing.Processes

include("Utils.jl")
include("LuxModel.jl")
include("LearningLoop.jl")
include("CaptureState.jl")
include("GraphSetup.jl")
include("Dynamics.jl")
include("Gradient.jl")

end # module IsingLearning
