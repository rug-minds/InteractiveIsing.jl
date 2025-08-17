module InteractiveIsing

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

# using QML
# export QML
using FileIO, ColorSchemes, Dates, JLD2, Random, Distributions, Observables, LinearAlgebra,
    StatsBase, LaTeXStrings, DataStructures, Preferences, GLMakie, SparseArrays, FFTW, ExprTools, UUIDs, DataStructures
using Images

using Processes
export Processes
import Processes: prepare

using PrecompileTools

#TEMP
using Revise

# import Plots as pl

#Temps
using SparseArrays, StaticArrays, LoopVectorization


export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end
abstract type AbstractIsingLayer{T,DIMS} <: AbstractIsingGraph{T} end



# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl

include("Utils/Utils.jl")


### DECLARED TYPES
@ForwardDeclare IsingGraph "Graphs"
@ForwardDeclare IsingLayer "Graphs/Layers"
# @ForwardDeclare Parameters "Graphs"
@ForwardDeclare IsingSim "Sim"
@ForwardDeclare SimLayout "Makie"

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

include("WeightFuncs.jl")
include("AdjList/AdjList.jl")

# @ForwardDeclare LayerMetaData "Graphs/Layers"
# @ForwardDeclare LayerArchitecture "Graphs/Layers"
include("MCAlgorithms/MCAlgorithms.jl")
# using .MCAlgorithms

include("Graphs/Graphs.jl")

include("Sim/Sim.jl")
include("Interaction/Interaction.jl")
include("Analysis/Analysis.jl")
include("Makie/Makie.jl")
include("GPlotting.jl")
include("Barebones.jl")


const ca1 = CompositeAlgorithm((LayeredMetropolis, Metropolis), (1,2))

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
