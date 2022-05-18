using LinearAlgebra, Distributions, Random, GLMakie, FileIO, QML, Observables, ColorSchemes, Images, DataFrames, CSV, CxxWrap
using BenchmarkTools
import Plots as pl
# Qt5QuickControls_jll, Qt5QuickControls2_jll,

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

# function timedFunctions(julia_display::JuliaCanvas)
#     Threads.@spawn let _
#         tfs = time()
#         # tfa = time()
#         while running[]
#             if time() - tfs > 0.034
#                     dispIsing(julia_display,g)
#                 tfs = time()
#             end
#             # if g.updates > g.size
#             #     magnetization($g,M,M_array)

#             #     g.updates = 0
#             #     # tfa = time()
#             # end
#             sleep(0.001)
#         end
        
#     end
# end


function persistentFunctions()
    # timedFunctions(julia_display)
    updateGraph()
end

# analysis_func = Threads.@spawn on(updates) do val
# analysis_func = on(updates) do val   
#     if updates[] > g.size
#         let _
#             magnetization(g,M,M_array)
#             updates[] = 0
#         end
#     end
#     yield()
# end

# Draw circle to state
# circleToStateQML(i,j) = Threads.@spawn circleToState(g,i,j,brushR[],brush[])

# Don't use for large state
function printG()
    println(g)
end


# dispIsingQML() = disp

addRandomDefectsQML() = Threads.@spawn addRandomDefects!(g,pDefects)

tempSweepQML() = Threads.@spawn CSV.write("sweepData.csv", dataToDF(tempSweep(g,TIs,M_array)))

@qmlfunction println
@qmlfunction addRandomDefectsQML
@qmlfunction initIsing
@qmlfunction printG
# @qmlfunction circleToStateQML
# @qmlfunction persistentFunctions
@qmlfunction tempSweepQML

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    # img = imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) , maxsize=g.N  )
    # buffer = reshape(buffer, size(img))
    # buffer = reinterpret(ARGB32, buffer)
    # buffer .= img
    return
  end

# showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
#                                                (Array{UInt32,1}, Int32, Int32))

# loadqml( qmlfile, obs =  pmap, showlatest = showlatest_cfunction); exec_async()
loadqml( qmlfile, obs =  pmap); exec_async() 






