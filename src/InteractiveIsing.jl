module InteractiveIsing
const modulefolder = @__DIR__


using   SparseArrays, StaticArrays, LoopVectorization,
        FileIO, ColorSchemes, Dates, JLD2, Random, 
        Distributions, Observables, LinearAlgebra,
        StatsBase, LaTeXStrings, DataStructures, 
        Preferences, GLMakie, SparseArrays, 
        FFTW, ExprTools, UUIDs, DataStructures, Images,
        Unitful
using PrecompileTools
using MacroTools



include("../deps/StatefulAlgorithms/src/StatefulAlgorithms.jl")

using .StatefulAlgorithms
export StatefulAlgorithms
import .StatefulAlgorithms: init, step!

include("TypeDefs.jl")
include("Utils/Utils.jl")

include("Topology/Topology.jl")

include("AdjList/AdjList.jl")
include("Proposals/Proposals.jl")
include("MCAlgorithms/MCAlgorithms.jl")

include("Graphs/Graphs.jl")
include("IndexPickers/IndexPickers.jl")

include("Sim/Sim.jl")

include("Interaction/Interaction.jl")

include("Analysis/Analysis.jl")

include("Makie/Makie.jl")

include("Images.jl")

include("Windows/Windows.jl")
include("Topology/TopologyDisplayExtensions.jl")

# Diagnostic local edit: disable InteractiveIsing precompile workload while
# isolating `StatefulAlgorithms.loop(::Repeat, ::NonGenerated)` codegen behavior.
# include("Precompile.jl")

# include("Barebones.jl")

@debug "InteractiveIsing module load complete"


end # module InteractiveIsing
