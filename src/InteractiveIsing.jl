module InteractiveIsing

@debug "Starting InteractiveIsing module load"

const mtimers = Timer[]
const mtasks = Task[]
const mptimers = Any[]

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

@debug "Loading dependencies"

# using QML
# export QML
using FileIO, ColorSchemes, Dates, JLD2, Random, Distributions, Observables, LinearAlgebra,
    StatsBase, LaTeXStrings, DataStructures, Preferences, GLMakie, SparseArrays, FFTW, ExprTools, UUIDs, DataStructures
using Images

@debug "Loading Processes module"

include("../deps/Processes/src/Processes.jl")

using .Processes
export Processes
import .Processes: prepare

@debug "Loading PrecompileTools and Revise"

using PrecompileTools

#TEMP
using Revise

@debug "Loading additional dependencies"

# import Plots as pl

#Temps
using SparseArrays, StaticArrays, LoopVectorization


export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end
abstract type AbstractIsingLayer{T,DIMS} end
abstract type AbstractLayerProperties end



# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl

@debug "Including Utils"
include("Utils/Utils.jl")
@debug "Utils loaded"


### DECLARED TYPES
@debug "Forward declaring types"
# @ForwardDeclare IsingGraph "Graphs/IsingGraph.jl"
# @ForwardDeclare IsingLayer "Graphs/Layers/IsingLayer.jl"
# @ForwardDeclare LayerProperties "Graphs/Layers/IsingLayer.jl"
# @ForwardDeclare Parameters "Graphs"
# @ForwardDeclare IsingSim "Sim"  # Commented out because IsingSim struct is commented out in source
# @ForwardDeclare SimLayout "Makie/SimLayout.jl"
abstract type AbstractSimLayout end
@debug "Forward declarations complete"

abstract type StateType end
struct Discrete <: StateType end
struct Continuous <: StateType end
struct Static <: StateType end #

# Base.isless(::Type{Continuous}, ::Type{Discrete}) = true
# Base.isless(::Type{Discrete}, ::Type{Continuous}) = false

# Base.isless(::Type{Discrete}, ::Type{Static}) = true
# Base.isless(::Type{Static}, ::Type{Discrete}) = false

# Base.isless(::Type{Continuous}, ::Type{Static}) = true
# Base.isless(::Type{Static}, ::Type{Continuous}) = false

# Base.isless(::Type{<:StateType}, ::Type{<:StateType}) = false
    
export Discrete, Continuous, Static

# Global RNG for module
const rng = MersenneTwister()

@debug "Including WeightFuncs"
include("WeightFuncs.jl")
@debug "WeightFuncs loaded"

@debug "Including Topology"
include("Topology/Topology.jl")
@debug "Topology loaded"

@debug "Including AdjList"
include("AdjList/AdjList.jl")
@debug "AdjList loaded"

# @ForwardDeclare LayerMetaData "Graphs/Layers"
# @ForwardDeclare LayerArchitecture "Graphs/Layers"
@debug "Including MCAlgorithms"
include("MCAlgorithms/MCAlgorithms.jl")
# using .MCAlgorithms
@debug "MCAlgorithms loaded"

@debug "Including Graphs"
include("Graphs/Graphs.jl")
@debug "Graphs loaded"

@debug "Including Sim"
include("Sim/Sim.jl")
@debug "Sim loaded"

@debug "Including Interaction"
include("Interaction/Interaction.jl")
@debug "Interaction loaded"

@debug "Including Analysis"
include("Analysis/Analysis.jl")
@debug "Analysis loaded"

@debug "Including Makie"
include("Makie/Makie.jl")
@debug "Makie loaded"

@debug "Including GPlotting"
include("GPlotting.jl")
@debug "GPlotting loaded"

@debug "Including Barebones"
include("Barebones.jl")
@debug "Barebones loaded"


# @debug "Creating CompositeAlgorithm"
const ca1 = CompositeAlgorithm((LayeredMetropolis, Metropolis), (1,2))
@debug "CompositeAlgorithm created"

@debug "InteractiveIsing module load complete"

# # PRECOMPILATION FUNCTION FOR FAST USAGE
# @setup_workload begin
#     GC.enable(false)

#     cg1 = IsingGraph(20, 20, type = Discrete)
#     cg3d = IsingGraph(20, 20, 20, type = Continuous)

#     @compile_workload begin
#         cwg = @WG "(dr) -> 1" NN=1
#         prepare(ca1, (;g = cg1))
#         genAdj!(cg1[1], cwg)
#         createProcess(cg1, ca1, lifetime = 10)
#         quit(cg1)
#         interface(cg1)
#         closeinterface()
  
#         prepare(ca1, (;g = cg3d))
#         interface(cg3d)
#         closeinterface()   
#         genAdj!(cg3d[1], cwg)
#         createProcess(cg3d, ca1, lifetime = 10)
#         fetch(process(cg3d))
#         quit(cg3d)
#         simulation |> reset!
        
#         GC.enable(true)
#     end
# end



end # module InteractiveIsing
