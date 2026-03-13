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
import .Processes: init

@debug "Loading PrecompileTools and Revise"

using PrecompileTools

#TEMP
# using Revise

@debug "Loading additional dependencies"

# import Plots as pl

#Temps
using SparseArrays, StaticArrays, LoopVectorization

include("TypeDefs.jl")
include("Utils/Utils.jl")


# Global RNG for module
const rng = MersenneTwister()

# @debug "Including WeightFuncs"
# include("WeightFuncs.jl")
# @debug "WeightFuncs loaded"

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

# @debug "Including Barebones"
# include("Barebones.jl")
# @debug "Barebones loaded"

@debug "InteractiveIsing module load complete"

# PRECOMPILATION FUNCTION FOR FAST USAGE
# @setup_workload begin
#     GC.enable(false)

#     @compile_workload begin
#         cg1 = IsingGraph(20, 20, type = Discrete)
#         cg3d = IsingGraph(20, 20, 20, type = Continuous)
        
#         cwg = @WG (dr) -> 1 NN=1
#         genAdj!(cg1[1], cwg)
#         createProcess(cg1, lifetime = 10)
#         quit(cg1)
#         interface(cg1)
#         # closeinterface()
  
#         interface(cg3d)
#         # closeinterface()   
#         genAdj!(cg3d[1], cwg)
#         createProcess(cg3d, lifetime = 10)
#         fetch(process(cg3d))
#         quit(cg3d)
        
#         GC.enable(true)
#     end
# end



end # module InteractiveIsing
