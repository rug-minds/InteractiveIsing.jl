"""
Re initialize simulation while running
"""
function reset!(sim::IsingSim)::Nothing
    for g in sim.gs
        reset!(g)
    end

    M(sim)[] = 0
    updates(sim, 0)

    branchSim(sim)
    return
end
export reset!

function annealing(sim, Ti, Tf, initWait = 30, stepWait = 5; Tstep = .5, T_it = Ti:Tstep:Tf, reInit = true, saveImg = true)
    # Reinitialize
    reInit && reset!(sim)

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
function startSim(sim; threads = 1, async = true, loadQML = true)
    if !started(sim)
        # Set sim started to true
        started(sim,true)

        # Spawn MCMC thread on layer 1
        # createProcess(sim, threads; gidx = 1)
        createProcess(sim; gidx = 1)

        # Load QML interface?
        # TODO: FIX THIS, remove these funtions
        # if loadQML
        #     # Register QML Functions
        #     # qmlFunctions(sim)
        #     # Set render loop global variable
        #     # setRenderLoop()

        #     # Load qml files, observables and functions
        #     # loadqml( qmlfile, obs = sim.pmap, showlatest = showlatest_cfunction)
        #     # Start interface in interactive mode or not
        #     if async
        #         exec_async()
        #     else
        #         exec()
        #     end
        # end
    # If no processes are started just create a new one
    elseif all(status(sim) .== :Terminated)
        createProcess(sim, threads; gidx = 1)
    else
        println("Simulation already started")
    end
        sim
end

function setCircR!(sim, r)
    brushR(sim)[] = r
    circ(sim, getOrdCirc(brushR(sim)[]))
end
export setCircR!

function setLayerIdx!(sim, layeridx)
    if layeridx < 1 || layeridx > nlayers(sim)[]
        println("Cannot choose layer idx smaller than 1 or larger than the number of layers")
        return
    end

    layerIdx(sim)[] = layeridx
end
export setLayerIdx!