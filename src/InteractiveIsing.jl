__precompile__()

module InteractiveIsing

using FileIO, Images, ColorSchemes, Dates, JLD, Random, Distributions, Observables, LinearAlgebra, DataFrames, 
    CSV, CxxWrap, LaTeXStrings

import Plots as pl

using QML
export QML

include("HelperFunctions.jl")
include("WeightFuncs.jl")
include("SquareAdj.jl")
include("IsingGraphs.jl")
include("Hamiltonians.jl")
include("IsingMetropolis.jl")
include("IsingSim.jl")
include("Interaction/Interaction.jl")
include("Analysis.jl")
include("GPlotting.jl")
include("IsingMagneticFields.jl")
include("Learning/IsingLearning.jl")

# Probably shouldn't be exported
export showlatest_cfunction
# Needs to be in init for pointer to img in IsingSim.jl to work
function __init__()
    global showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
                                               (Array{UInt32,1}, Int32, Int32))
end


end # module InteractiveIsing
