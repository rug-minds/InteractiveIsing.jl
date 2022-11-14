#= 
Magnetic field stuff
=#



# Set M field for given indexes
export setMIdxs!
function setMIdxs!(sim,idxs,strengths)
    g = sim.g
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    g.d.mlist[idxs] = strengths
    setSimHType(sim, :MagField => true)
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
    setMIdxs!(sim,[1:g.size;],reshape(transpose(m_matr),g.size));
    return
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

export setMFuncRepeating!
function setMFuncRepeating!(sim, func::Function, t_step = .1)
    repeating = Ref(true)

    function loopM()
        t = 0
        while repeating[]
            newfunc(;x,y) = func(;x,y,t)
            setMFunc!(sim,newfunc)
            t+= t_step
        end
        remM!(sim)
    end
    errormonitor( Threads.@spawn loopM() )
    println("Press any button to cancel")
    run(`stty raw`)
    read(stdin, Char)
    run(`stty cooked`);
    repeating[] = false
    return
end

# Removes magnetic field
export remM!
function remM!(sim)
    g = sim.g
    g.d.mlist = zeros(Float32, g.size)
    setSimHType(sim, :MagField => false)
end

# Plot magnetic field
export plotM
function plotM(sim)
    g = sim.g
    imagesc(permutedims(reshape(g.d.mlist, g.N,g.N)))
end