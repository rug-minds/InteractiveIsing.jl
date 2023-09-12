# __precompile__(false)
# __precompile__(true)

module InteractiveIsing

macro time_sm(using_expr)
    expr = quote end
    push!(expr.args, :(println("Timing submodules")))
    for mod in using_expr.args
        modulename = mod.args[1]
        push!(expr.args, :(@time string($modulename) using $(modulename)))
    end
    return esc(expr)
end

const modulefolder = @__DIR__

# using QML
# export QML

using FileIO, ColorSchemes, Dates, JLD2, Random, Distributions, Observables, LinearAlgebra,
    StatsBase, LaTeXStrings, DataStructures, Preferences, GLMakie, SparseArrays
# using Images
using PrecompileTools

#TEMP
using Revise

import Plots as pl

#Temps
using SparseArrays, LoopVectorization

import Base: getindex, setindex!, length, iterate, isless, push!, resize!, size

export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end
abstract type AbstractIsingLayer{T} <: AbstractIsingGraph{T} end

abstract type PeriodicityType end
struct Periodic <: PeriodicityType end
struct NonPeriodic <: PeriodicityType end
struct NoPeriodicity <: PeriodicityType end

export PeriodicityType, Periodic, NonPeriodic

# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl
branchSim(sim) = refreshSim(sim)
export branchSim
include("HelperFiles/HelperFiles.jl")


@ForwardDeclare IsingGraph "IsingGraphs"
@ForwardDeclare IsingLayer "IsingGraphs/Layers"

include("WeightFuncs.jl")
include("AdjList/AdjList.jl")

@ForwardDeclare IsingSim "Sim"

include("Hamiltonians/Hamiltonians.jl")
include("IsingGraphs/IsingGraphs.jl")

include("Sim/Sim.jl")
include("Interaction/Interaction.jl")
include("Analysis/Analysis.jl")
include("Makie/Makie.jl")
include("GPlotting.jl")

# include("Learning/IsingLearning.jl")

# Probably doesn't need to be exported
# export showlatest_cfunction
# Needs to be in init for pointer to img in IsingSim.jl to work
# function __init__()
#     global showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
#                                                (Array{UInt32,1}, Int32, Int32))
# end

# PRECOMPILATION FUNCTION FOR FAST USAGE
# @setup_workload begin
#     csim = IsingSim(
#         20,
#         20,
#         continuous = true, 
#         weighted = true;
#         colorscheme = ColorSchemes.winter
#     );

#     cg = csim(false)

#     @compile_workload begin
#         addLayer!(cg, 20, 20)

#         setcoords!(cg[1])
#         setcoords!(cg[2], z = 1)

#         connectLayers!(cg, 1, 2, (;dr, _...) -> 1, 1)

#         #Plotting correlation length and GPU kernel
#         plotCorr(cg[2], dodisplay = false, save = false)

#         setSpins!(cg[1], 1, 1, true, false)

#         drawCircle(cg[1], 1, 1, 1, clamp = true)

#         path = saveGraph(cg, savepref = false)

#         loadGraph(path)        

#         # TODO: Deletefile
#     end
# end



end # module InteractiveIsing
