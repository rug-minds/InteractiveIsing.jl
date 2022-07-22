""" Parameters """
function setRenderLoop()
    ENV["QSG_RENDER_LOOP"] = "basic"
end

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Main.qml")
# qmlfile = joinpath(dirname(Base.source_path()), "qml", "TSweepWindow/Tsweep.qml")

# const pmap = JuliaPropertyMap(
#     "isRunning" => isRunning,
#     "gSize" => gSize,
#     "NIs" => NIs, 
#     "TIs" => TIs, 
#     "JIs" => JIs, 
#     "brush" => brush,
#     "brushR" => brushR,
#     "M" => M,
#     "upf" => upf,
#     "imgSize" => imgSize,
#     "analysisRunning" => analysisRunning
# )

const pmap = JuliaPropertyMap()

#Observables etc
const shouldRun = Observable(true)
pmap["shouldRun"] = shouldRun
const isRunning = Ref(true)

const gSize = Observable(graphSize)
pmap["gSize"] = gSize

const NIs = Observable(gSize[])
pmap["NIs"] = NIs
const TIs = Observable(initTemp)
pmap["TIs"] = TIs
const JIs = Observable(1.0)
pmap["JIs"] = JIs

const brush = Observable(0)
pmap["brush"] = brush

const brushR = Observable( Int(round(NIs[]/10)) )
pmap["brushR"] = brushR
const circ  = Observable(getOrdCirc(brushR[])) 
pmap["circ"] = circ

const M = Observable(0.0)
pmap["M"] = M

const analysisRunning = Observable(false)
pmap["analysisRunning"] = analysisRunning

# Not elegant
const M_array = Ref(zeros(Real,60))
# const M_array = zeros(Int32,avg_window)


# Locking updating mag
const updatingMag = Ref(false)


# Counting MMC updates
const updates = Ref(0)
# Updates per frame
const upf = Observable(0)
pmap["upf"] = upf

# Locking thread
const updatingUpf = Ref(false)

# Graph
if continuous
    const g = CIsingGraph(
            NIs[], 
            weighted = weighted,
            weightFunc = weighted ? weightFunc : defaultIsingWF
        )
else
    const g = IsingGraph(
        NIs[], 
        weighted = weighted,
        weightFunc = weighted ? weightFunc : defaultIsingWF
    )
end

# Image
const img = Ref(gToImg(g))
const imgSize = Observable(size(img[]))
pmap["imgSize"] = imgSize

# Locking img updating thread
const updatingImg = Ref(false)

const getERef = Ref(HFunc)



