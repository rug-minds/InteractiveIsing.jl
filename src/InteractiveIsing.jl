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
    StatsBase, LaTeXStrings, DataStructures, Preferences, GLMakie, SparseArrays, FFTW
using Images
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
# struct NoPeriodicity <: PeriodicityType end

export PeriodicityType, Periodic, NonPeriodic

# Restart MCMC loop to define new Hamiltonian function
# Is needed for fast execution if part of hamiltonian doesn't need to be checked
# Should be in IsingSim.jl

include("Utils/Utils.jl")


### DECLARED TYPES
@ForwardDeclare IsingGraph "IsingGraphs"
@ForwardDeclare IsingLayer "IsingGraphs/Layers"
@ForwardDeclare IsingSim "Sim"

abstract type StateType end
struct Discrete <: StateType end
struct Continuous <: StateType end
struct Static <: StateType end

Base.isless(::Type{Continuous}, ::Type{Discrete}) = true
Base.isless(::Type{Discrete}, ::Type{Continuous}) = false

Base.isless(::Type{Discrete}, ::Type{Static}) = true
Base.isless(::Type{Static}, ::Type{Discrete}) = false

Base.isless(::Type{Continuous}, ::Type{Static}) = true
Base.isless(::Type{Static}, ::Type{Continuous}) = false

Base.isless(::Type{<:StateType}, ::Type{<:StateType}) = false
    
export Discrete, Continuous, Static


include("WeightFuncs.jl")
include("AdjList/AdjList.jl")


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
@setup_workload begin
    GC.enable(false)
    cg = simulate(20,20, type = Continuous, start = false, run = false, disp = false, noinput = true)
    quit(cg)
    _sim = sim(cg)

    @compile_workload begin

        addLayer!(cg, 20, 20, type = Discrete)

        setcoords!(cg[1])
        setcoords!(cg[2], z = 1)
        cwg = @WG "(dr) -> 1" NN=1

        genAdj!(cg[1], cwg)
        genAdj!(cg[1],cg[2], cwg)

        #Plotting correlation length and GPU kernel
        plotCorr(cg[2], dodisplay = false, save = false)

        setSpins!(cg[1], 1, 1, true, false)

        drawCircle(cg[1], 1, 1, 1, clamp = true)

        path = saveGraph(cg, savepref = false)

        close.(values(timers(_sim)))
        loadGraph(path)       


        closeinterface()
        reset!(simulation)
        GC.enable(true)
    end
end



end # module InteractiveIsing
