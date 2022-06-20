ENV["QSG_RENDER_LOOP"] = "basic"

include("Sim.jl")
include("WeightFuncs.jl")

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Ising.qml")

# Global Variables

    # User Parameters

    const periodic = true

    # Is simulation weighted
    const weighted = true
    # How many nearest neighbors, hard cutoff
    const NN = 1
    const weightFunc(r) = randWeights(r, dist = Normal(1,0.01))


#Observables
const running = Observable(true)
const gSize = Observable(512)
const NIs = Observable(gSize[])
const TIs = Observable(1.0)
const JIs = Observable(1.0)
const isPaused = Observable(false) 
const brush = Observable(0)
const brushR = Observable( Int(round(NIs[]/10)) )
const circ  = Observable(getOrdCirc(brushR[]))
const M = Observable(0.0)


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
    periodic = periodic, 
    weighted = weighted,
    NN = NN,
    weightFunc = weighted ? weightFunc : defIsing
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







