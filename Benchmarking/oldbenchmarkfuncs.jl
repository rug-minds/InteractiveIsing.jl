
function benchmarkFunc(sim,iterations)
    g = sim.g
    TIs = sim.TIs
    getE = g.d.hFuncRef[]

    # Defining argumentless functions here seems faster.
    # Offset large function into files for clearity
    function updateMonteCarloIsingC!()
        T = TIs[]
        @inline function deltE(efac,newstate,oldstate)
            return efac*(newstate-oldstate)
        end
    
        @inline function sampleCState()
            Float32(2*(rand()-.5))
        end
    
        beta = T>0 ? 1/T : Inf
        
        idx = rand(ising_it(g))

        oldstate = g.state[idx]
    
        efactor = getE(g,idx, oldstate)
    
        newstate = sampleCState()
        
        Ediff = deltE(efactor,newstate,oldstate)
        if (Ediff < 0 || rand() < exp(-beta*Ediff))
            @inbounds g.state[idx] = newstate 
        end
    end

    function updateMonteCarloIsingD!()
        T = TIs[]
        @inline function deltE(Estate)
            return -2*Estate
        end
        
        beta = T>0 ? 1/T : Inf
                
        idx = rand(ising_it(g))

        Estate = g.state[idx]*getE(g,idx)
        
        if (Estate >= 0 || rand() < exp(-beta*deltE(Estate)))
            @inbounds g.state[idx] *= -1
        end
        
    end

    if typeof(g) == IsingGraph{Int8}
        isingUpdate! = updateMonteCarloIsingD!
    else
        isingUpdate! = updateMonteCarloIsingC!
    end
    sim.updates = 0

    t_start = time()

    for _ in 1:(iterations)
        isingUpdate!()
        sim.updates += 1
    end
    t_end = time()

    return t_end-t_start
end



#Other stuff
function updateMonteCarloIsingC2!(sim)
    g = sim.g
    T = sim.TIs[]
    getE = g.d.hFuncRef[]
    @inline function deltE(efac,newstate,oldstate)
        return efac*(newstate-oldstate)
    end

    @inline function sampleCState()
        Float32(2*(rand()-.5))
    end

    beta = T>0 ? 1/T : Inf
    
    idx = rand(ising_it(g))

    oldstate = g.state[idx]

    efactor = getE(g,idx, oldstate)

    newstate = sampleCState()
    
    Ediff = deltE(efactor,newstate,oldstate)
    if (Ediff < 0 || rand() < exp(-beta*Ediff))
        @inbounds g.state[idx] = newstate 
    end
end

function updateMonteCarloIsingD2!(sim)
    g = sim.g
    T = sim.TIs[]
    getE = g.d.hFuncRef[]

    @inline function deltE(Estate)
        return -2*Estate
    end
    
    beta = T>0 ? 1/T : Inf
            
    idx = rand(ising_it(g))

    Estate = g.state[idx]*getE(g,idx)
    
    if (Estate >= 0 || rand() < exp(-beta*deltE(Estate)))
        @inbounds g.state[idx] *= -1
    end
    
end

export benchmarkFunc2
function benchmarkFunc2(sim)
    g = sim.g
    TIs = sim.TIs

    # Defining argumentless functions here seems faster.
    # Offset large function into files for clearity
    

    if typeof(g) == IsingGraph{Int8}
        isingUpdate! = updateMonteCarloIsingD2!
    else
        isingUpdate! = updateMonteCarloIsingC2!
    end
    sim.updates = 0
    t_start = time()
    for _ in 1:(200000*30*10)
        isingUpdate!(g)
        sim.updates += 1
    end
    t_end = time()

    return t_end-t_start
end


export benchmarkFunc3
function benchmarkFunc3(sim)
    g = sim.g
    TIs = sim.TIs
    getE = (g, idx, state = g.state[idx]) -> HWeightedFunc(g,idx,state) + clampE(idx)

    # Defining argumentless functions here seems faster.
    # Offset large function into files for clearity
    function updateMonteCarloIsingC!()
        T = TIs[]
        @inline function deltE(efac,newstate,oldstate)
            return efac*(newstate-oldstate)
        end
    
        @inline function sampleCState()
            Float32(2*(rand()-.5))
        end
    
        beta = T>0 ? 1/T : Inf
        
        idx = rand(ising_it(g))

        oldstate = g.state[idx]
    
        efactor = getE(g,idx, oldstate)
    
        newstate = sampleCState()
        
        Ediff = deltE(efactor,newstate,oldstate)
        if (Ediff < 0 || rand() < exp(-beta*Ediff))
            @inbounds g.state[idx] = newstate 
        end
    end

    function updateMonteCarloIsingD!()
        T = TIs[]
        @inline function deltE(Estate)
            return -2*Estate
        end
        
        beta = T>0 ? 1/T : Inf
                
        idx = rand(ising_it(g))

        Estate = g.state[idx]*getE(g,idx)
        
        if (Estate >= 0 || rand() < exp(-beta*deltE(Estate)))
            @inbounds g.state[idx] *= -1
        end
        
    end

    if typeof(g) == IsingGraph{Int8}
        isingUpdate! = updateMonteCarloIsingD!
    else
        isingUpdate! = updateMonteCarloIsingC!
    end
    sim.updates = 0

    t_start = time()

    for _ in 1:(200000*30*10)
        isingUpdate!()
        sim.updates += 1
    end
    t_end = time()

    return t_end-t_start
end

function efactest(g::AbstractIsingGraph,idx)::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor
end

function getFac(conn)
    @inbounds -connW(conn)*g.state[connIdx(conn)]
end

function efactest2(g::AbstractIsingGraph,idx)::Float32 
    return @inbounds reduce(+,getFac.(g.adj[idx]))
end