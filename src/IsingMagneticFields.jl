#= 
Magnetic field stuff
=#


export setMIdxs!
""" 
Set the magnetic field using two vectors, one gives the indexes of the positions,
the other gives the magnetic field strength for the corresponding index
"""
function setMIdxs!(sim, layer ,idxs,strengths)
    g = sim.layers[layer]
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    g.d.mlist[idxs] .= strengths
    setSimHType!(sim, :MagField => true)
end

export setMFunc!
""" 
Set the magnetic field based on a given function of x and y.
Function needs to be specified as a julia anonymous functions that needs to
have the named arguments x and y. The syntax to define an anonymous function
is (;x,y) -> f(x,y)
"""
function setMFunc!(sim, layer, func::Function)
    g = sim.layers[layer]
    m_matr = Matrix{Float32}(undef,g.N,g.N)
    for y in 1:g.N
        for x in 1:g.N
            m_matr[y,x] = func(;x,y)
        end
    end
    setMIdxs!(sim,[1:g.size;],reshape(transpose(m_matr),g.size));
    return
end

export setMFuncTimed!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
"""
function setMFuncTimed!(sim, layer, func::Function, interval = 5, t_step = .2)
    g = sim.layers[layer]
    for t in 0:t_step:interval
        newfunc(;x,y) = func(;x,y,t)
        setMFunc!(sim,newfunc)
        sleep(t_step)
    end
end

export setMFuncRepeating!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
which keeps repeating until a button is pressed
"""
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

export remM!
"""
Removes magnetic field
"""
function remM!(sim, layer)
    g = sim.layers[layer]
    g.d.mlist = zeros(Float32, g.size)
    setSimHType!(sim, :MagField => false)
end

export plotM
""" 
Plot the magnetic field
"""
function plotM(sim, layer)
    g = sim.layers[layer]
    imagesc(permutedims(reshape(g.d.mlist, g.N,g.N)))
end