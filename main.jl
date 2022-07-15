include("Sim.jl")
include("WeightFuncsCustom.jl")


"""
User Variables
"""
const runsim = true
const continuous = false
const graphSize = 512
const weighted = true
const weightFunc = radialWF

# setAddDist!(weightFunc, Normal(0,0.3))

const initTemp = 1.

include("Params.jl")

# Start Simulation
if runsim
    startSim()
    loadqml( qmlfile, obs =  pmap, showlatest = showlatest_cfunction); exec_async()
end