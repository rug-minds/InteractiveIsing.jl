include("IsingParams.jl")
include("Obs.jl")
include("IsingSim.jl")

@forward IsingSim Obs
@forward IsingSim IsingParams params
# @setterGetter IsingSim]
export gs
@inline gs(sim) = sim.gs
export layers
@inline layers(sim) = sim.layers
@inline layers(sim, idx) = vecvecIdx(layers(sim), idx)
export graph
@inline graph(sim, graphidx = 1) = gs(sim)[graphidx]
# @inline shouldRun(sim) = sim.shouldRun
# @inline isRunning(sim) = sim.isRunning
# @inline isRunning(sim,val) = sim.isRunning = val
@inline M_array(sim) = sim.M_array
export currentLayer
@inline currentLayer(sim) = vecvecIdx(layers(sim), activeLayer(sim)[])

include("QML.jl")
include("Loop.jl")
include("TimedFunctions.jl")
include("User.jl")

# For plotting
const img =  Ref(zeros(RGB{Float64},1,1))

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reinterpret(ARGB32, buffer)
    buffer .= @view permutedims(img[])[1:end]
    return
end
export showlatest

# Pauses sim and waits until paused
function pauseSim(sim)
    println("Pausing sim")

    shouldRun(sim,false)

    while isRunning(sim)
        yield()
    end

    return true
end
export pauseSim

function unpauseSim(sim)
    println("Pausing sim")
    isRunning(sim) && return

    shouldRun(sim, true)

    while !isRunning(sim)
        yield()
    end

    return true
end


# """ REPL FUNCTIONS FOR DEBUGGING """

# # # Draw circle to state
# circleToStateQML(i,j,clamp=false) = Threads.@spawn circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])
# circleToStateREPL(i,j, clamp = false) = circleToState(g, circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])

# function tempSweepQMLRepl(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true); analysisRunning[] = true; tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, savelast = true) end
