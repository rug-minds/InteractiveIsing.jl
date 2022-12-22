#= All functions and vars that are needed for QML to work =#

qmlfile = joinpath( @__DIR__, ".." , "qml", "Main.qml")


"""
Set the qml setRenderLoop environment variables
"""
function setRenderLoop()
    ENV["QSG_RENDER_LOOP"] = "basic"
end

"""
Register all qml functions.
Should work since the macro only calls a function
and all the local variables are actually variables that live in the sim object outside of the function
"""
function qmlFunctions(sim::IsingSim)
    s_brush = brush(sim)
    s_Temp = Temp(sim)
    s_M_array = M_array(sim)
    s_M = M(sim)
    s_brushR = brushR(sim)

    # Locks for ensuring spawned functions are not created
    # if old one is still running
    
    # Clean up this system
    # Maybe with a dict
    updatingImg = sim.updatingImg
    updatingUpf = sim.updatingUpf
    updatingMag = sim.updatingMag

    s_analysisRunning = analysisRunning(sim)

    @qmlfunction println

    # All functions that are run from the QML Timer
    function timedFunctions()
        layer = currentLayer(sim)
        checkImgSize(sim, layer, glength(layer), gwidth(layer), qmllength(sim), qmllength(sim))
        spawnOne(updateImg, updatingImg, "UpdateImg", sim)
        spawnOne(updatesPerFrame, updatingUpf, "", sim)
        spawnOne(magnetization, updatingMag, "", sim)
    end
    @qmlfunction timedFunctions


    # Add percentage of defects to lattice
    addRandomDefectsQML(pDefects) = addRandomDefects!(sim, currentLayer, pDefects)
    @qmlfunction addRandomDefectsQML

    # Initialize isinggraph and display
    function initIsing()
        reInitSim(sim) 
    end
    @qmlfunction initIsing

    # Draw circle to state
    circleToStateQML(i,j,clamp=false) = errormonitor(Threads.@spawn circleToState(sim, currentLayer(sim), i, j, s_brush[]; clamp, imgsize = size(img[])))
    @qmlfunction circleToStateQML

    # Sweep temperatures and record magnetization and correlation lengths
    # Make an interface for this
    function tempSweepQML(TI = s_Temp[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true)
        if !g.d.defects
            corrF = sampleCorrPeriodic
        else
            corrF = sampleCorrPeriodicDefects
        end
        # Catching error doesn't work like this, why?
        Threads.@spawn errormonitor(tempSweep(sim, currentLayer(sim) , s_Temp, s_M_array; TI, TF, TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, s_analysisRunning=s_analysisRunning, corrF))
    end
    @qmlfunction tempSweepQML


    # Save a new circle with size s_brushR[]
    function newCirc()
        circ(sim, getOrdCirc(s_brushR[]))
    end
    @qmlfunction newCirc

    # Save an image of the graph
    saveGImgQML() = saveGImg(sim, currentLayer(sim))
    @qmlfunction saveGImgQML

    function setTemp(temp)
        sim.params.Temp = temp
    end
    @qmlfunction setTemp

    function toggleSimRunning()
        togglePauseSim(sim)
    end
    @qmlfunction toggleSimRunning

    function changeLayer(inc)
        setLayerIdx!(sim, layerIdx(sim)[] + inc)
        newR = round(min(size(currentLayer(sim))...) / 10)

        setCircR!(sim, newR) 
    end
    @qmlfunction changeLayer
    
end
