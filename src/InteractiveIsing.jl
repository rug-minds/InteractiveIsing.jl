module InteractiveIsing
const modulefolder = @__DIR__


using   SparseArrays, StaticArrays, LoopVectorization,
        FileIO, ColorSchemes, Dates, JLD2, Random, 
        Distributions, Observables, LinearAlgebra,
        StatsBase, LaTeXStrings, DataStructures, 
        Preferences, GLMakie, SparseArrays, 
        FFTW, ExprTools, UUIDs, DataStructures, Images
using PrecompileTools


include("../deps/Processes/src/Processes.jl")

using .Processes
export Processes
import .Processes: init, step!

include("TypeDefs.jl")
include("Utils/Utils.jl")

include("Topology/Topology.jl")

include("AdjList/AdjList.jl")

include("MCAlgorithms/MCAlgorithms.jl")

include("Graphs/Graphs.jl")

include("Sim/Sim.jl")

include("Interaction/Interaction.jl")

include("Analysis/Analysis.jl")

include("Makie/Makie.jl")

include("Images.jl")

# include("Barebones.jl")

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
