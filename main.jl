
include("Sim.jl")
include("CustomWF.jl")


"""
User Variables
"""
const graphSize = 512
const weighted = false
const weightFunc = defaultIsingWF

const initTemp = 1.


"""
Simulations Parameters
Don't have to be touched

"""

ENV["QSG_RENDER_LOOP"] = "basic"

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Main.qml")
# qmlfile = joinpath(dirname(Base.source_path()), "qml", "TSweepWindow/Tsweep.qml")


#Observables
const running = Observable(true)
const gSize = Observable(graphSize)
const NIs = Observable(gSize[])
const TIs = Observable(initTemp)
const JIs = Observable(1.0)
const isPaused = Observable(false) 
const brush = Observable(0)
const brushR = Observable( Int(round(NIs[]/10)) )
const circ  = Observable(getOrdCirc(brushR[])) 
const M = Observable(0.0)

# Not elegant
const M_array = Ref(zeros(Int32,60))
# const M_array = zeros(Int32,avg_window)


# Locking updating mag
const updatingMag = Ref(false)


# Counting MMC updates
const updates = Ref(0)
# Updates per frame
const upf = Observable(0)
# Locking thread
const updatingUpf = Ref(false)

# Graph
const g = IsingGraph(
    NIs[], 
    weighted = weighted,
    weightFunc = weighted ? weightFunc : defaultIsingWF
    )

# Image
const img = Ref(gToImg(g))
# Locking img updating thread
const updatingImg = Ref(false)


# Basically a dict of all properties
const pmap = JuliaPropertyMap(
    "running" => running,
    "gSize" => gSize,
    "NIs" => NIs, 
    "TIs" => TIs, 
    "JIs" => JIs, 
    "isPaused" => isPaused, 
    "brush" => brush,
    "brushR" => brushR,
    "M" => M,
    "upf" => upf
)

@qmlfunction timedFunctions
@qmlfunction println
@qmlfunction addRandomDefectsQML
@qmlfunction initIsing
@qmlfunction circleToStateQML
@qmlfunction startSim
@qmlfunction tempSweepQML
@qmlfunction newCirc

# Start Simulation
startSim()
loadqml( qmlfile, obs =  pmap, showlatest = showlatest_cfunction); exec_async()






