
""" Magnetic field stuff """

# Restart MCMC loop to define new Hamiltonian function
function branchSim(sim)
    sim.shouldRun[] = false 
    while sim.isRunning[]
        sleep(.1)
    end
    sim.shouldRun[] = true
end

# Sets magnetic field and branches simulation
function setGHFunc!(sim, prt = true)
    g = sim.g
    if !g.d.weighted
        if !g.d.mactive
            g.d.hFuncRef = Ref(HFunc)
            if prt
                println("Set HFunc")
            end
        else
            g.d.hFuncRef = Ref(HMagFunc)
            if prt
                println("Set HMagFunc")
            end
        end
    else
        if !g.d.mactive
            g.d.hFuncRef = Ref(HWeightedFunc)
            if prt
                println("Set HWeightedFunc")
            end
        else
            g.d.hFuncRef = Ref(HWMagFunc)
            if prt
                println("Set HWMagFunc")
            end
        end
    end

    branchSim(sim)
end

# Set M field for given indexes
export setMIdxs!
function setMIdxs!(sim,idxs,strengths)
    g = sim.g
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    g.d.mactive = true
    g.d.mlist[idxs] = strengths
    setGHFunc!(sim, false)
end

# Insert func(;x,y)
export setMFunc!
function setMFunc!(sim,func::Function)
    g = sim.g
    m_matr = Matrix{Float32}(undef,g.N,g.N)
    for y in 1:g.N
        for x in 1:g.N
            m_matr[y,x] = func(;x,y)
        end
    end
    setMIdxs!(sim,[1:g.size;],reshape(transpose(m_matr),g.size))
end

# Set a time dependent magnetic field function f(;x,y,t)
export setMFuncTimed!
function setMFuncTimed!(sim,func::Function, interval = 5, t_step = .2)
    g = sim.g
    for t in 0:t_step:interval
        newfunc(;x,y) = func(;x,y,t)
        setMFunc!(sim,newfunc)
        sleep(t_step)
    end
end

# Removes magnetic field
export remM!
function remM!(sim)
    g = sim.g
    g.d.mactive = false
    setGHFunc!(sim, false)
end