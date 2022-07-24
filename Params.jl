""" Parameters """


qmlfile = joinpath(dirname(Base.source_path()), "qml", "Main.qml")

const pmap = JuliaPropertyMap()

#Observables etc
const shouldRun = Observable(true)
pmap["shouldRun"] = shouldRun
const isRunning = Ref(true)

const gSize = Observable(graphSize)
pmap["gSize"] = gSize

const TIs = Observable(initTemp)
pmap["TIs"] = TIs

const brush = Observable(0)
pmap["brush"] = brush

const brushR = Observable( Int(round(gSize[]/10)) )
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
            gSize[], 
            weighted = weighted,
            weightFunc = weighted ? weightFunc : defaultIsingWF
        )
else
    const g = IsingGraph(
        gSize[], 
        weighted = weighted,
        weightFunc = weighted ? weightFunc : defaultIsingWF
    )
end

# Image
# const img = Ref(gToImg(g))
# const imgSize = Observable(size(img[]))
# pmap["imgSize"] = imgSize

# Locking img updating thread
const updatingImg = Ref(false)



