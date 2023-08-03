include("IsingParams.jl")
include("Obs.jl")
include("Processes.jl")
include("Pausing.jl")
include("IsingSim.jl")



include("QML.jl")
include("Loop.jl")
include("timedFunctions.jl")
include("User.jl")

# For plotting
const img =  Ref(zeros(RGB{Float64},1,1))

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reinterpret(ARGB32, buffer)
    buffer .= @view permutedims(img[])[1:end]
    return
end
export showlatest


# """ REPL FUNCTIONS FOR DEBUGGING """

# # # Draw circle to state
# circleToStateQML(i,j,clamp=false) = Threads.@spawn circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])
# circleToStateREPL(i,j, clamp = false) = circleToState(g, circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])

# function tempSweepQMLRepl(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true); analysisRunning[] = true; tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, savelast = true) end
