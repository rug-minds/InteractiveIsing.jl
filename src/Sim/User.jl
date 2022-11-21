"""
Re initialize simulation while running
"""
function reInitSim(sim)
    g = sim.g
    g.state = typeof(g) == IsingGraph{Int8} ? initRandomState(g.size) : initRandomCState(g.size)
    editHType!(g, :MagField => false, :Defects => false, :Clamp => false)
    g.d.defectBools = [false for x in 1:g.size]
    g.d.defectList = []
    g.d.aliveList = [1:g.size;]
    g.d.mlist = zeros(Float32, g.size)

    sim.M[] = 0
    sim.updates = 0

    branchSim(sim)
end


# Pauses sim and waits until paused
export pauseSim
function pauseSim(sim)
    sim.shouldRun[] = false

    while sim.isRunning[]
        yield()
    end

    return true
end

export unpauseSim
function unpauseSim(sim)
    sim.isRunning && return

    sim.shouldRun[] = true

    while !sim.isRunning
        yield()
    end

    return true
end

function annealing(sim, Ti, Tf, initWait = 30, stepWait = 5; Tstep = .5, T_it = Ti:Tstep:Tf, reInit = true, saveImg = true)
    # Reinitialize
    reInit && initIsing()

    # Set temp and initial wait
    TIs[] = Ti
    sleep(initWait)
    
    for temp in T_it
        TIs[] = temp
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
function startSim(sim; async)
    if !sim.started
        sim.started = true
        setRenderLoop()
        qmlFunctions(sim)
        runSim(sim; async)
    else
        error("Simulation already running")
    end

end
