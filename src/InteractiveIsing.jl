# __precompile__(false)
__precompile__(true)

module InteractiveIsing

using FileIO, Images, ColorSchemes, Dates, JLD, Random, Distributions, Observables, LinearAlgebra, DataFrames, 
    CSV, CxxWrap, StatsBase, Metal, QML, Qt6ShaderTools_jll, LaTeXStrings, Plots
using Revise

import Plots as pl


using QML
export Qt6ShaderTools_jll
export QML

# Remove this
using BenchmarkTools

import Base: getindex, setindex!, length, iterate, isless, push!, resize!, size


# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl
branchSim(sim) = refreshSim(sim)
export branchSim

include("HelperFunctions.jl")
include("WeightFuncs.jl")
include("SquareAdj.jl")

include("Hamiltonians/Hamiltonians.jl")

import Base: size
@ForwardDeclare IsingSim "Sim"
include("IsingGraphs/IsingGraphs.jl")

# include("IsingMetropolis.jl")
include("Sim/Sim.jl")
include("Interaction/Interaction.jl")
include("Interaction/IsingMagneticFields.jl")
include("Interaction/Clamping.jl")
include("Analysis/Analysis.jl")
include("GPlotting.jl")

# include("Learning/IsingLearning.jl")

# Probably doesn't need to be exported
export showlatest_cfunction
# Needs to be in init for pointer to img in IsingSim.jl to work
function __init__()
    global showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
                                               (Array{UInt32,1}, Int32, Int32))
end


end # module InteractiveIsing
