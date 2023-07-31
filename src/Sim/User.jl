"""
Re initialize simulation while running
"""
function reset!(sim)::Nothing
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
        createProcess(sim, threads; gidx = 1)

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

"""

"""
function addLayer!(sim::IsingSim, glength, gwidth; gidx = 1, weightfunc = defaultIsingWF, periodic = true, type = typeof(state(gs(sim)[gidx])).parameters[1])
    #pause sim
    lock(processes(sim))
    pauseSim(sim, ignore_lock = true)
    # add layer to graph
    addLayer!(gs(sim)[gidx], glength, gwidth; periodic, weightfunc, type)
    #update number of layers
    nlayers(sim)[] += 1
    # unpause sim
    unpauseSim(sim, ignore_lock = true)
    unlock(processes(sim))
end

function removeLayer!(sim::IsingSim, layeridx; gidx = 1)
    pauseSim(sim, block = true)
    
    # If the slected layer is after the layer to be removed, decrement layerIdx
    if layerIdx(sim)[] >= layeridx && layerIdx(sim)[] > 1
        layerIdx(sim)[] -= 1
    end

    removeLayer!(gs(sim)[gidx], layeridx)
    nlayers(sim)[] -= 1

    unpauseSim(sim)
end
export addLayer!
export removeLayer!

function connectLayersFull(layer1::IsingLayer, layer2::IsingLayer)
    Threads.@threads for idx1 in 1:length(state(layer1))
        sidx = 1
        for idx2 in 1:length(state(layer2))
            sidx = addWeightDirected!(layer1, layer2, idx1, idx2, 1; sidx) + 1
        end
    end

    Threads.@threads for idx1 in 1:length(state(layer2))
        sidx = 1
        for idx2 in 1:length(state(layer1))
            sidx = addWeightDirected!(layer2, layer1, idx1, idx2, 1; sidx) + 1
        end
    end
end

connectLayersFull(g, layeridx1::Integer, layeridx2::Integer) = connectLayersFull(layers(g)[layeridx1], layers(g)[layeridx2])
export connectLayersFull


# Move this away


function removeEntries(vec, startidx, endidx, transform = identity)
    newstate = Vector{typeof(vec).parameters[1]}(undef, length(vec) - (endidx - startidx + 1) )

    @inbounds newstate[1:(startidx-1)]    .=  @inbounds vec[1:(startidx-1)]
    @inbounds newstate[startidx:end]      .=  @inbounds transform.(vec[(endidx+1):end])

    return newstate
end




function removeAdjEntries(adj::Vector{T}, idxs) where T
    newadj = Vector{T}(undef, length(adj) - length(idxs))
    mask = ones(Bool, length(adj))
    mask[idxs] .= 0
    newadj = adj[mask]

    for idx in idxs
        for conn in adj[idx]
            println(conn)
            newConns = Vector{Conn}(undef, length(newadj[connIdx(conn)]) - 1)
            newconnIdx = 1
            for otherconn in newadj[connIdx(conn)]
                println("Otherconn", otherconn)
                if connIdx(otherconn) == idx
                    continue
                end
                newConns[newconnIdx] = otherconn
                newconnIdx += 1
            end
            newadj[connIdx(conn)] = newConns
        end
    end

    return newadj
end
export removeAdjEntries