"""
Re initialize simulation while running
"""
function reInitSim(sim)::Nothing
    for layer in sim.layers
        reinitIsingGraph!(layer)
    end

    M(sim)[] = 0
    updates(sim, 0)

    branchSim(sim)
end

export togglePauseSim
function togglePauseSim(sim)
    if isRunning(sim)
        pauseSim(sim)
    else
        unpauseSim(sim)
    end
end

function annealing(sim, Ti, Tf, initWait = 30, stepWait = 5; Tstep = .5, T_it = Ti:Tstep:Tf, reInit = true, saveImg = true)
    # Reinitialize
    reInit && initIsing()

    # Set temp and initial wait
    Temp(sim)[] = Ti
    sleep(initWait)
    
    for temp in T_it
        Temp(sim)[] = temp
        sleep(stepWait)
        if saveImg
            save(File{format"PNG"}("Images/Annealing/Ising T$temp.PNG"), img[])
        end
    end
end

export startSim
# Set the render loop,
# Define qml functions
# Start graph update and qml interface
function startSim(sim; async = true)
    if !started(sim)
        started(sim,true)
        setRenderLoop()
        qmlFunctions(sim)
        runSim(sim; async)
    else
        error("Simulation already running")
    end

end
