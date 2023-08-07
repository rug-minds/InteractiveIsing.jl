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
# """
function qmlFunctions(sim::IsingSim)
    s_brush = brush(sim)
    s_Temp = Temp(sim)
    s_M_array = M_array(sim)
    s_M = M(sim)
    s_brushR = brushR(sim)

    s_analysisRunning = analysisRunning(sim)

    @qmlfunction println

    # All functions that are run from the QML Timer
    timedFunctionsQML() = timedFunctions(sim)

    @qmlfunction timedFunctionsQML


    # Add percentage of defects to lattice
    addRandomDefectsQML(pDefects) = (layer = currentLayer(sim); try addRandomDefects!(layer, pDefects); catch(error); rethrow(); end)
    @qmlfunction addRandomDefectsQML

    # Initialize isinggraph and display
    function initIsing()
        reset!(sim) 
    end
    @qmlfunction initIsing

    # Draw circle to state
    circleToStateQML(i,j,clamp=false) = (errormonitor(Threads.@spawn drawCircle(currentLayer(sim), i, j, s_brush[]; clamp)))
    @qmlfunction circleToStateQML

    # Sweep temperatures and record magnetization and correlation lengths
    # Make an interface for this
    function tempSweepQML(TI = s_Temp[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true)
        # Catching error doesn't work like this, why?
            
        errormonitor(
            Threads.@spawn begin 
                    try 
                        tempSweep(sim; TI, TF, TStep, dpoints , dpointwait, stepwait, equiwait, saveImg)
                    catch error
                        display(error)
                        analysisRunning(sim)[] = false
                    end 
            end
        )
    end
    @qmlfunction tempSweepQML


    # Save a new circle with size s_brushR[]
    function newCirc()
        circ(sim, getOrdCirc(s_brushR[]))
    end
    @qmlfunction newCirc

    # Save an image of the graph
    saveGImgQML() = saveGImg(currentLayer(sim))
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
        layerName(sim)[] = name(currentLayer(sim))
    end
    @qmlfunction changeLayer
    
    # TODO: Check this
    #This is not correct?
    function pauseUnpause()
        memory(sim)["Procstatus"] = [process.status for process in processes(sim)]
    end

    function setLayerName(str)
        layer = currentLayer(sim)
        name(layer,str)
    end
    @qmlfunction setLayerName
end
