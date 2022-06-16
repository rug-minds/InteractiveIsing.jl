using LinearAlgebra, Distributions, Random, GLMakie, FileIO, QML, Observables, ColorSchemes, Images, DataFrames, CSV, CxxWrap
using BenchmarkTools
import Plots as pl
# Qt5QuickControls_jll, Qt5QuickControls2_jll,


include("IsingGraphs.jl")
using .IsingGraphs

include("Interaction/Interaction.jl")
using .Interaction
include("Analysis.jl")
using .Analysis

include("MCMC.jl") 
include("GPlotting.jl")
# using .GPlotting

function insertShift(vec::Vector{T}, el::T) where T
    newVec = Vector{T}(undef, length(vec))
    newVec[1:(end-1)] = vec[2:end]
    newVec[end] = el
    return newVec
end

# Spawn function thread, but only if such a thread is not already active
function spawnOne(f::Function, spawned::Ref{Bool}, args...)
    # Run function, when thread is finished mark it as not running
    function runOne(func::Function, spawned::Ref{Bool}, args...)
        func(args...)
        spawned[] = false
        GC.safepoint()
    end

    # Mark as running, then spawn thread
    if !spawned[]
        spawned[] = true
        # Threads.@spawn runOne(f,spawned)
        runOne(f,spawned, args...)
    end
end


# Main loop for for MCMC
function updateGraph()
        while running[]
            if !isPaused[] # if sim not paused
                updateMonteCarloQML!(g,TIs[],JIs[])
                updates[] += 1
            else
                sleep(0.2)
            end
            GC.safepoint()
        end
end

function updateGraphREPL()
    while running[]
    
        if !isPaused[] # if sim not paused
            updateMonteCarloQML!(g,TIs[],JIs[])
            updates[] += 1
        else
            sleep(0.2)
        end
    end
end

function startSim()
    Threads.@spawn updateGraph()
end

"""Timed Functions"""
# Updating image of graph
function updateImg()
    img[] = gToImg(g,gSize[])
end

# Track number of updates per frame
let avgWindow = 30, updateWindow = zeros(Int64,avgWindow), frames = 0
    global function updatesPerFrame()
        updateWindow = insertShift(updateWindow,updates[])
        if frames > avgWindow
            upf[] = round(sum(updateWindow)/avgWindow)
            frames = 0
        end
        frames += 1
        updates[] = 0
    end
end

""" QML FUNCTIONS """

function timedFunctions()
    spawnOne(updateImg, updatingImg)
    spawnOne(updatesPerFrame, updatingUpf)
    spawnOne(magnetization, updatingMag, g,M)
end


# addRandomDefectsQML(pDefects) = Threads.@spawn addRandomDefects!(g,pDefects)
addRandomDefectsQML(pDefects) = addRandomDefects!(g,pDefects)

# Initialize isinggraph and display
function initIsing()
    reInitGraph!(g) 
    M[] = 0
end

# Draw circle to state
circleToStateQML(i,j) = Threads.@spawn circleToState(g,circ[],i,j,brush[])
circleToStateREPL(i,j) = circleToState(g,circ[],i,j,brush[])

# Make an interface for this
tempSweepQML(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 ) = Threads.@spawn CSV.write("sweepData.csv", dataToDF(tempSweep(g,TIs,M_array, TI=TI,TF=TF,TStep=TStep, dpoints = dpoints, dpointwait= dpointwait, stepwait = stepwait, equiwait=equiwait)))

# New circle
function newCirc()
    circ[] = getOrdCirc(brushR[])
end

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reshape(buffer, size(img[]))
    buffer = reinterpret(ARGB32, buffer)
    buffer .= img[]
    return
end

showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
                                               (Array{UInt32,1}, Int32, Int32))

    