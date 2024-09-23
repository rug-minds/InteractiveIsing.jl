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
    StatsBase, LaTeXStrings, DataStructures, Preferences, GLMakie, SparseArrays, FFTW, ExprTools, UUIDs
using Images
using PrecompileTools

#TEMP
using Revise

# import Plots as pl

#Temps
using SparseArrays, StaticArrays, LoopVectorization

# import Base: getindex, setindex!, length, iterate, isless, push!, resize!, size

export AbstractIsingGraph
abstract type AbstractIsingGraph{T} end
abstract type AbstractIsingLayer{T,DIMS} <: AbstractIsingGraph{T} end

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
@ForwardDeclare IsingGraph "Graphs"
@ForwardDeclare IsingLayer "Graphs/Layers"
# @ForwardDeclare IsingParameters "Graphs"
@ForwardDeclare IsingSim "Sim"
@ForwardDeclare SimLayout "Makie"

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

# Global RNG for module
const rng = MersenneTwister()

include("WeightFuncs.jl")
include("AdjList/AdjList.jl")

include("MCAlgorithms/MCAlgorithms.jl")
# using .MCAlgorithms

include("Hamiltonians/Hamiltonians.jl")
include("Graphs/Graphs.jl")

include("Sim/Sim.jl")
include("Interaction/Interaction.jl")
include("Analysis/Analysis.jl")
include("Makie/Makie.jl")
include("GPlotting.jl")
include("Barebones.jl")

# PRECOMPILATION FUNCTION FOR FAST USAGE
@setup_workload begin
    GC.enable(false)

    cg = IsingGraph(20, 20, type = Discrete)
    simulate(cg, start = false)
    _sim = sim(cg)

    @compile_workload begin

        # addLayer!(cg, 20, 20)

        # setcoords!(cg[1])
        # setcoords!(cg[2], z = 1)
        cwg = @WG "(dr) -> 1" NN=1

        genAdj!(cg[1], cwg)
        # genAdj!(cg[1],cg[2], cwg)

        # #Plotting correlation length and GPU kernel
        # plotCorr(cg[2], dodisplay = false, save = false)

        # setSpins!(cg[1], 1:3, 1, true, false)

        # drawCircle(cg[1], 1, 1, 1, clamp = true)

        # path = saveGraph(cg, savepref = false)

        close.(values(timers(_sim)))
        # loadGraph(path)       

        closeinterface()

        # w = LayerWindow(cg[1])
        # closewindow(w)
        # w = createAnalysisWindow(cg[1], MT_panel, tstep = 0.01)
        # closewindow(w)
        # w = createAnalysisWindow(cg[1], MB_panel, tstep = 0.01)
        # closewindow(w)
        # w = createAnalysisWindow(cg, χₘ_panel, Tχ_panel, shared_interval = 1/500, tstep = 0.01);
        # closewindow(w)

        reset!(simulation)
        GC.enable(true)
    end
end



end # module InteractiveIsing
