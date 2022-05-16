using LinearAlgebra, Distributions, Random, GLMakie, FileIO, QML, Qt5QuickControls_jll, Qt5QuickControls2_jll, Observables, ColorSchemes, Images, DataFrames, CSV
using BenchmarkTools
import Plots as pl


include("ising_graph.jl")
include("ising_update.jl")
include("square_adj.jl")
include("plotting.jl")
include("interaction.jl")
include("analysis.jl")

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Ising.qml")

#Observables
running = Observable(true)
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

# brushR = Observable(2)
# circle = Observable(getCircle(0,0,brushR))

# Global Variables
g = IsingGraph(NIs[])




# Basically a dict of all properties
pmap = JuliaPropertyMap(
    "running" => running,
    "NIs" => NIs, 
    "TIs" => TIs, 
    "JIs" => JIs, 
    "isPaused" => isPaused, 
    "pDefects" => pDefects,
    "brush" => brush,
    "brushR" => brushR,
    "M" => M,
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
        while running[]
        
            if !isPaused[] # if sim not paused
                updateMonteCarloQML!(g,TIs[],JIs[])
                updates[] += 1
            else
                sleep(0.2)
            end
            
        end
    end
end

function timedFunctions(julia_display::JuliaDisplay)
    Threads.@spawn let _
        tfs = time()
        # tfa = time()
        while running[]
            if time() - tfs > 0.0333
                dispIsing(julia_display,g)
                tfs = time()
            end
            # if g.updates > g.size
            #     magnetization($g,M,M_array)

            #     g.updates = 0
            #     # tfa = time()
            # end
            
            sleep(0.001)
        end
        
    end
end


function persistentFunctions(julia_display::JuliaDisplay)
    timedFunctions(julia_display)
    updateGraph()
end

analysis_func = Threads.@spawn on(updates) do val
    if updates[] > g.size
        let _
            magnetization($g,M,M_array)
            updates[] = 0
        end
    end
end

# Draw circle to state
circleToStateQML(i,j) = Threads.@spawn circleToState(g,i,j,brushR[],brush[])

# Don't use for large state
function printG()
    println(g)
end

addRandomDefectsQML() = Threads.@spawn addRandomDefects!(g,pDefects)

tempSweepQML() = Threads.@spawn CSV.write("sweepData.csv", dataToDF(tempSweep(g,TIs,M_array) , 100))

@qmlfunction println
@qmlfunction addRandomDefectsQML
@qmlfunction initIsing
@qmlfunction printG
@qmlfunction circleToStateQML
@qmlfunction persistentFunctions
@qmlfunction tempSweepQML


loadqml( qmlfile, obs =  pmap); exec_async() 





