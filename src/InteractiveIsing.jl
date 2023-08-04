# __precompile__(false)
__precompile__(true)

module InteractiveIsing

const modulefolder = @__DIR__

using FileIO, Images, ColorSchemes, Dates, JLD2, Random, Distributions, Observables, LinearAlgebra, 
    CxxWrap, StatsBase, LaTeXStrings
using Plots
using PrecompileTools
using Revise

import Plots as pl

include("./QML.jl")
using QML, Qt6ShaderTools_jll
# export Qt6ShaderTools_jll
export QML

import Base: getindex, setindex!, length, iterate, isless, push!, resize!, size


# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl
branchSim(sim) = refreshSim(sim)
export branchSim

include("HelperFiles/HelperFiles.jl")
include("WeightFuncs.jl")
include("SquareAdj.jl")

@ForwardDeclare IsingSim "Sim"

include("Hamiltonians/Hamiltonians.jl")
include("IsingGraphs/IsingGraphs.jl")

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

# PRECOMPILATION FUNCTION FOR FAST USAGE
@setup_workload begin
    csim = IsingSim(
        20,
        20,
        continuous = true, 
        weighted = true;
        colorscheme = ColorSchemes.winter
    );
    cg = csim(false)
    @compile_workload begin
        addLayer!(csim, 20, 20)

        # # name them l1, l2, l3 ...
        @enumeratelayers layers(cg) 2

        setcoords!(l1)
        setcoords!(l2, z = 1)

        clampImg!(cg, 1, "examples/smileys.jpg")
        connectLayers!(cg, 1, 2, (;dr, _...) -> 1, 1)

        #Plotting correlation length and GPU kernel
        plotCorr(correlationLength(l1)...)

        setSpins!(l1, 1, 1, true, false)

        
    end
end



end # module InteractiveIsing
