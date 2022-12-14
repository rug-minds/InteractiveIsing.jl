"""
Re initialize simulation while running
"""
function reInitSim(sim)::Nothing
    for g in sim.gs
        reinitIsingGraph!(g)
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
function startSim(sim; async = true, loadQML = true)
    if !started(sim)
        # Set sim started to true
        started(sim,true)

        # Spawn MCMC thread on layer 1
        errormonitor(Threads.@spawn updateSim(sim, 1))

        # Load QML interface?
        if loadQML
            # Register QML Functions
            qmlFunctions(sim)
            # Set render loop global variable
            setRenderLoop()

            # Load qml files, observables and functions
            loadqml( qmlfile, obs = sim.pmap, showlatest = showlatest_cfunction)
            # Start interface in interactive mode or not
            if async
                exec_async()
            else
                exec()
            end
        end
    else
        error("Simulation already running")
    end

end

function setCircR(sim, r)
    brushR(sim)[] = r
    circ(sim, getOrdCirc(brushR(sim)[]))
end
export setCircR