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
    return
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
function startSim(sim; threads = 1, async = true, loadQML = true)
    if !started(sim)
        # Set sim started to true
        started(sim,true)

        # Spawn MCMC thread on layer 1
        println("Created a process")
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
    else
        error("Simulation already running")
    end

end

function setCircR!(sim, r)
    brushR(sim)[] = r
    circ(sim, getOrdCirc(brushR(sim)[]))
end
export setCircR!

function setLayerIdx!(sim, layeridx)
    if layeridx < 1 || layeridx > nlayers(sim)[]
        println("Cannot choose layer idx smaller than 0 or larger than the number of layers")
        return
    end

    layerIdx(sim)[] = layeridx
end
export setLayerIdx!

function addLayer!(sim, gidx, glength, gwidth, weightfunc = defaultIsingWF)
    pauseSim(sim)
    glayers = layers(sim)[gidx]
    g = gs(sim)[gidx]
    oldlength = nStates(g)

    # Resize underlying graphs 
    resizeG!(g, glength*gwidth)

    # Save layer parameters
    layerparams = []
    for layer in glayers
        push!(layerparams, tuple(start(layer), size(layer)...))
    end

    startidx = layerparams[end][1] + layerparams[end][2]*layerparams[end][3]
    push!(layerparams, (startidx, glength, gwidth))

    # Make new layers
    newlayers = Vector{IsingLayer}(undef, length(glayers) + 1)
    for layeridx in 1:(length(glayers) + 1)
        newlayers[layeridx] = IsingLayer(g, layerparams[layeridx]...)
    end

    layers(sim)[gidx] = newlayers
    
    fillAdjList!(newlayers[end], glength, gwidth, weightfunc)
    
    unpauseSim(sim)

    nlayers(sim)[] += 1
    return
end

function removeLayer!(sim, gidx, layeridx)
    pauseSim(sim)
    g = gs(sim)[gidx]
    layervec = layers(sim)[gidx]
    layer = layers(sim)[gidx][layeridx]

    # Remove the states from the IsingGraph
    state(g, removeStates(state(g), start(layer), start(layer) + nStates(layer) - 1))

    # Removing defects
    ndefects = collectNumDefects(layervec)
    newdefects = remPartitionAscendingList(defectList(g), ndefects, layeridx, nStates(layer))
    defectList(g, newdefects)

    # Fixing aliveList
    nalives = nStates.(layervec) .- ndefects
    newalives = remPartitionAscendingList(aliveList(g), nalives, layeridx, nStates(layer))
    aliveList(g, newalives)

    unpauseSim(sim)
end
export addLayer!

function removeEntries(vec, startidx, endidx, transform = identity)
    newstate = Vector{typeof(vec).parameters[1]}(undef, length(vec) - (endidx - startidx + 1) )

    @inbounds newstate[1:(startidx-1)]    .=  @inbounds vec[1:(startidx-1)]
    @inbounds newstate[startidx:end]      .=  @inbounds transform.(vec[(endidx+1):end])

    return newstate
end

"""
Takes an ascending list that is partitioned into parameters
Removes one of the partitions, and shifts al elements after the partition
down by an amount of the maxsize of the partition
|part1|part2|part3| -> |part1|part3 .- maxsize(part2)|
"""
function remPartitionAscendingList(list, npartitions, partitionidx, maxSize)
    nElementsToRemove = npartitions[partitionidx]
    startidx = 1 + sum(npartitions[1:(partitionidx-1)])
    endidx = startIdx + nElementsToRemove
    removeEntries(list, startidx, endidx, x -> x - maxSize)
end

"""
From vector of layers for one underlying graph
collect the number of defects
"""
function collectNumDefects(layers)
    defectvec = Vector(undef, length(layers))
    for (idx,layer) in enumerate(layers)
        defectvec[idx] = ndefects(layer)
    end
    return defectvec
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