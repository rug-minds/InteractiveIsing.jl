ENV["QSG_RENDER_LOOP"] = "basic"

include("Sim.jl")

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Ising.qml")

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


# Global Variables
const g = IsingGraph(NIs[])

# Image
const img = Ref(gToImg(g))
# Locking img updating thread
const updatingImg = Ref(false)

# For threads, don't make new ones if already funning



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
# loadqml( qmlfile, obs =  pmap, ); exec_async() 

loadqml( qmlfile, obs =  pmap, showlatest = showlatest_cfunction); exec_async()







