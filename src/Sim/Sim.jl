include("IsingSim.jl")
include("QML.jl")
include("Loop.jl")
include("TimedFunctions.jl")
include("User.jl")

# For plotting
const img =  Ref(zeros(RGB{Float64},1,1))

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reshape(buffer, size(img[]))
    buffer = reinterpret(ARGB32, buffer)
    buffer .= transpose(img[])
    return
end


export runSim
# Spawn graph update thread and load qml interface
function runSim(sim; load = true, async = true)
    # showlatest_cfunction = showlatesteval(sim)
    Threads.@spawn errormonitor(updateGraph(sim))
    if load
        loadqml( qmlfile, obs = sim.pmap, showlatest = showlatest_cfunction)
        if async
            exec_async()
        else
            exec()
        end
    end
end


# """ REPL FUNCTIONS FOR DEBUGGING """

# # # Draw circle to state
# circleToStateQML(i,j,clamp=false) = Threads.@spawn circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])
circleToStateREPL(i,j, clamp = false) = circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])

function tempSweepQMLRepl(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true); analysisRunning[] = true; tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, savelast = true) end
