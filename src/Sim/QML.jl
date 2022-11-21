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
    g = sim.g
    circ = sim.circ
    brush = sim.brush
    TIs = sim.TIs
    M_array = sim.M_array
    M = sim.M
    brushR = sim.brushR

    # Locks for ensuring spawned functions are not created
    # if old one is still running
    
    # Clean up this system
    # Maybe with a dict
    updatingImg = sim.updatingImg
    updatingUpf = sim.updatingUpf
    updatingMag = sim.updatingMag

    analysisRunning = sim.analysisRunning

    @qmlfunction println

    # All functions that are run from the QML Timer
    function timedFunctions()
        spawnOne(updateImg, updatingImg, "UpdateImg", sim)
        spawnOne(updatesPerFrame, updatingUpf, "", sim)
        spawnOne(magnetization, updatingMag, "", sim)
    end
    @qmlfunction timedFunctions


    # Add percentage of defects to lattice
    addRandomDefectsQML(pDefects) = addRandomDefects!(sim, g,pDefects)
    @qmlfunction addRandomDefectsQML

    # Initialize isinggraph and display
    function initIsing()
        reInitSim(sim) 
    end
    @qmlfunction initIsing

    # Draw circle to state
    circleToStateQML(i,j,clamp=false) = errormonitor(Threads.@spawn circleToState(sim, g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1]))
    @qmlfunction circleToStateQML

    # Sweep temperatures and record magnetization and correlation lengths
    # Make an interface for this
    function tempSweepQML(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true)
        if !g.d.defects
            corrF = sampleCorrPeriodic
        else
            corrF = sampleCorrPeriodicDefects
        end
        # Catching error doesn't work like this, why?
        Threads.@spawn errormonitor(tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, corrF))
    end
    @qmlfunction tempSweepQML


    # Save a new circle with size brushR[]
    function newCirc()
        circ[] = getOrdCirc(brushR[])
    end
    @qmlfunction newCirc

    # Save an image of the graph
    saveGImgQML() = saveGImg(g)
    @qmlfunction saveGImgQML
end
