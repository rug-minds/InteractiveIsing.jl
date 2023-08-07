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