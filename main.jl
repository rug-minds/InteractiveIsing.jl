push!(LOAD_PATH, pwd())

include("Sim.jl")
include("WeightFuncsCustom.jl")


"""
User Variables
"""
const runsim = true
const continuous = true
const graphSize = 100
const weighted = true
const weightFunc = isingNN2

# setAddDist!(weightFunc, Normal(0,0.3))

const initTemp = 1.

include("Params.jl")
setRenderLoop()
# Start Simulation
if runsim
    startSim(g)
    loadqml( qmlfile, obs =  pmap, showlatest = showlatest_cfunction); exec_async()
end