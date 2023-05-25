#= 
Magnetic field stuff
=#


export setMIdxs!
""" 
Set the magnetic field using two vectors, one gives the indexes of the positions,
the other gives the magnetic field strength for the corresponding index
"""
function setMIdxs!(layer, strengths; idxs = 1:length(strengths))

    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    mlist(layer)[idxs] .= strengths[1:end]

    setSType!(graph(layer), :Magfield => true)
end

export setMFunc!
""" 
Set the magnetic field based on a given function of x and y.
Function needs to be specified as a julia anonymous functions that needs to
have the named arguments x and y. The syntax to define an anonymous function
is (;x,y) -> f(x,y)
"""
function setMFunc!(layer, func::Function)
    m_matr = Matrix{Float32}(undef,glength(layer),gwidth(layer))
    for x in 1:gwidth(layer)
        for y in 1:glength(layer)
            m_matr[y,x] = func(;x,y)
        end
    end
    setMIdxs!(layer, m_matr);
    return
end

export setMFuncTimed!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
"""
function setMFuncTimed!(layer, func::Function, interval = 5, t_step = .2)
    for t in 0:t_step:interval
        newfunc(;x,y) = func(;x,y,t)
        setMFunc!(layer, newfunc)
        sleep(t_step)
    end
end

export setMFuncRepeating!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
which keeps repeating until a button is pressed
"""
function setMFuncRepeating!(layer, func::Function, t_step = .1)
    repeating = Ref(true)

    function loopM()
        t = 0
        while repeating[]
            newfunc(;x,y) = func(;x,y,t)
            setMFunc!(layer, newfunc)
            t+= t_step
            sleep(t_step)
        end
        remM!(layer)
    end
    errormonitor( Threads.@spawn loopM() )
    println("Press any button to cancel")
    run(`stty raw`)
    read(stdin, Char)
    run(`stty cooked`);
    repeating[] = false
    return
end

function setMFuncTimer!(layer, func::Function, t_step = .2) 
    t = 0
    while t < interval
        newfunc(;x,y) = func(;x,y,t)
        setMFunc!(layer, newfunc)
        sleep(t_step)
        t += t_step
    end
    remM!(layer)
end

export remM!
"""
Removes magnetic field
"""
function remM!(layer)
    mlist(layer) .= Float32(0)
    setSType!(layer, :Magfield => false)
end

export plotM
""" 
Plot the magnetic field
"""
function plotM(sim; g = currentLayer(sim))
    imagesc(mlist(g))
end