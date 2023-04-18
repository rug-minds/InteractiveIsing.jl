#= 
Magnetic field stuff
=#


export setMIdxs!
""" 
Set the magnetic field using two vectors, one gives the indexes of the positions,
the other gives the magnetic field strength for the corresponding index
"""
function setMIdxs!(sim, strengths; idxs = 1:length(strengths), g = currentLayer(sim))
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end
    mlist(g)[iterator(g)] .= strengths[1:end]
    setSimHType!(sim, :MagField => true)
end

export setMFunc!
""" 
Set the magnetic field based on a given function of x and y.
Function needs to be specified as a julia anonymous functions that needs to
have the named arguments x and y. The syntax to define an anonymous function
is (;x,y) -> f(x,y)
"""
function setMFunc!(sim, func::Function; g = currentLayer(sim))
    m_matr = Matrix{Float32}(undef,glength(g),gwidth(g))
    for x in 1:gwidth(g)
        for y in 1:glength(g)
            m_matr[y,x] = func(;x,y)
        end
    end
    setMIdxs!(sim, m_matr; g);
    return
end

export setMFuncTimed!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
"""
function setMFuncTimed!(sim, func::Function, interval = 5, t_step = .2; g = currentLayer(sim))
    for t in 0:t_step:interval
        newfunc(;x,y) = func(;x,y,t)
        setMFunc!(sim, newfunc; g)
        sleep(t_step)
    end
end

export setMFuncRepeating!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
which keeps repeating until a button is pressed
"""
function setMFuncRepeating!(sim, func::Function, t_step = .1;  g = currentLayer(sim))
    repeating = Ref(true)

    function loopM()
        t = 0
        while repeating[]
            newfunc(;x,y) = func(;x,y,t)
            setMFunc!(sim, newfunc; g)
            t+= t_step
        end
        remM!(sim, 1)
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
function remM!(sim; g = currentLayer(sim))
    mlist(g) .= Float32(0)
    setSimHType!(sim, :MagField => false)
end

export plotM
""" 
Plot the magnetic field
"""
function plotM(sim; g = currentLayer(sim))
    imagesc(mlist(g))
end