module IsingLearning

using LuxCore
using Random: AbstractRNG
using SparseArrays
using InteractiveIsing
using InteractiveIsing: state, adj, setparam!, getparam, setSpins!, nStates
using InteractiveIsing.Processes
using DataStructures
using ChainRules, ChainRulesCore

import LuxCore: initialparameters, initialstates
import InteractiveIsing.Processes: init, step!

include("Utils.jl")
include("LuxModel.jl")
include("LearningLoop.jl")
include("CaptureState.jl")
include("GraphSetup.jl")
include("Dynamics.jl")

end # module IsingLearning
