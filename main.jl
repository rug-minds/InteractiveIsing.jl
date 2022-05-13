using LinearAlgebra, BenchmarkTools, Distributions, Random, GLMakie, FileIO, QML, Qt5QuickControls_jll, Qt5QuickControls2_jll, Observables, ColorSchemes, Images, CSV
import Plots as pl


include("ising_graph.jl")
include("ising_update.jl")
include("square_adj.jl")
include("plotting.jl")
include("interaction.jl")
include("analysis.jl")

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Ising.qml")

#Observables
NIs = Observable(512)
TIs = Observable(1.0)
JIs = Observable(1.0)
pDefects = Observable(0) #percentage of defects to be added
isPaused = Observable(false) 
brush = Observable(0)
brushR = Observable( Int(round(NIs[]/10)) )
M = Observable(0.0)
M_array = []

updates = Observable(0)
updates_frame = Observable(0)

# brushR = Observable(2)
# circle = Observable(getCircle(0,0,brushR))

# Global Variables
g = IsingGraph(NIs[])




# Basically a dict of all properties
pmap = JuliaPropertyMap(
    "NIs" => NIs, 
    "TIs" => TIs, 
    "JIs" => JIs, 
    "isPaused" => isPaused, 
    "pDefects" => pDefects,
    "brush" => brush,
    "brushR" => brushR,
    "M" => M,
    "uframe" => updates_frame
)

"""QML Functions"""
# Initialize isinggraph and display
function initIsing()
    reInitGraph!(g) 
    M[] = 0
end

# Main loop for QML
function updateGraph()
    Threads.@spawn let _
        # Main loop
        while true
        
            if !isPaused[] # if sim not paused
                # updates += 1
                updateMonteCarloQML!(g,TIs[],JIs[])
            else
                sleep(0.2)
            end

            
        end
    end
end

function updateScreen(julia_display::JuliaDisplay)
    Threads.@spawn let _
        while true
            dispIsing(julia_display,g)
            # updates_frame[] = updates
            # updates[] = 0
            Threads.@spawn magnetization(g,M,M_array)
            sleep(0.0333) # ~30fps at most
        end
    end
end

function persistentFunctions(julia_display::JuliaDisplay)
    updateScreen(julia_display)
    updateGraph()
end

# Draw circle to state
circleToStateQML(i,j) = Threads.@spawn circleToState(g,i,j,brushR[],brush[])

# Don't use for large state
function printG()
    println(g)
end

addRandomDefectsQML() = Threads.@spawn addRandomDefects!(g,pDefects)

@qmlfunction println
@qmlfunction addRandomDefectsQML
@qmlfunction initIsing
@qmlfunction printG
@qmlfunction circleToStateQML
@qmlfunction persistentFunctions


loadqml( qmlfile, obs =  pmap); exec_async() 




