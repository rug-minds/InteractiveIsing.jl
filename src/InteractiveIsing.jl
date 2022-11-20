__precompile__(false)

module InteractiveIsing

using FileIO, Images, ColorSchemes, Dates, JLD, Random, Distributions, Observables, LinearAlgebra, DataFrames, 
    CSV, CxxWrap, LaTeXStrings

import Plots as pl

using QML
export QML

# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl
function branchSim(sim)
    if sim.shouldRun[]
        sim.shouldRun[] = false 
        while sim.isRunning[]
            yield()
        end
        sim.shouldRun[] = true;
    end
end
export branchSim

export HType
struct HType{Symbs, Vals} end



include("HelperFunctions.jl")
include("WeightFuncs.jl")
include("SquareAdj.jl")

include("Hamiltonians.jl")
using .Hamiltonians
export Hamiltonians

include("IsingGraphs.jl")

include("SetEls.jl")

# include("IsingMetropolis.jl")
include("IsingSim.jl")
include("Interaction/Interaction.jl")
include("Analysis.jl")
include("GPlotting.jl")
include("IsingMagneticFields.jl")

# include("Learning/IsingLearning.jl")

# Probably doesn't need to be exported
export showlatest_cfunction
# Needs to be in init for pointer to img in IsingSim.jl to work
function __init__()
    global showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
                                               (Array{UInt32,1}, Int32, Int32))
end


end # module InteractiveIsing
