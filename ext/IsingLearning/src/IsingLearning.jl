module IsingLearning

using LuxCore
using Random: AbstractRNG, MersenneTwister
using SparseArrays
using LinearAlgebra: diag

using DataStructures
using ChainRules, ChainRulesCore

import LuxCore: initialparameters, initialstates

# include local interactivesising
include(joinpath(@__DIR__, "..", "..","..", "src", "InteractiveIsing.jl"))
import .InteractiveIsing.StatefulAlgorithms: init, step!
using .InteractiveIsing
using .InteractiveIsing: state, adj, setparam!, getparam, setSpins!, nStates
using .InteractiveIsing.StatefulAlgorithms

include("Utils.jl")
include("Tools/LearningProcessTools.jl")
include("LuxModel.jl")
include("LearningLoop.jl")
include("CaptureState.jl")
include("GraphSetup.jl")
include("ReadoutClamping.jl")
include("Dynamics.jl")
include("Gradient.jl")
include("ThreadedMNISTLoop.jl")

end # module IsingLearning
