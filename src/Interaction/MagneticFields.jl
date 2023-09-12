#= 
Magnetic field stuff
=#


export setBIdxs!
""" 
Set the magnetic field using two vectors, one gives the indexes of the positions,
the other gives the magnetic field strength for the corresponding index
"""
function setBIdxs!(layer, idxs::Vector, strengths::Vector)

    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    bfield(layer)[idxs] .= strengths[1:end]

    setSType!(graph(layer), :Magfield => true)
end
setBIdxs!(layer, strengths) = setBIdxs!(layer, 1:length(strengths), strengths)

""" 
Set the magnetic field based on a given function of x and y.
Function needs to be specified as a julia anonymous functions that needs to
have the named arguments x and y. The syntax to define an anonymous function
is (;x,y) -> f(x,y)
"""
function setBFunc!(layer, func::Function)
    b_mat = bfield(layer)
    for y in 1:gwidth(layer)
        for x in 1:glength(layer)
            b_mat[x,y] = func(;x,y)
        end
    end
    setSType!(graph(layer), :Magfield => true)
    return
end
export setBFunc!


export setBFuncTimed!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
"""
function setBFuncTimed!(layer, func::Function, interval = 5, t_step = .2)
    for t in 0:t_step:interval
        newfunc(;x,y) = func(;x,y,t)
        setBFunc!(layer, newfunc)
        sleep(t_step)
    end
end

export setBFuncRepeating!
"""
Set a time dependent magnetic field function (;x,y,t) -> f(x,y,t)
which keeps repeating until a button is pressed
"""
function setBFuncRepeating!(layer, func::Function, t_step = .1)
    repeating = Ref(true)

    function loopM()
        t = 0
        while repeating[]
            ti = time()
            newfunc(;x,y) = func(;x,y,t)
            setBFunc!(layer, newfunc)
            t+= t_step
            sleep(max(0, t_step - (time() - ti)))
        end
        remB!(layer)
    end
    errormonitor( Threads.@spawn loopM() )
    println("Press any button to cancel")
    run(`stty raw`)
    read(stdin, Char)
    run(`stty cooked`);
    repeating[] = false
    return
end

"""

"""
function setBFuncTimer!(layer, func::Function, t_step = .2) 
    t = Ref(0.)
    repeatfunc(timer) = begin
        ti = time()
        newfunc(;x,y) = func(;x,y, t=t[])
        setBFunc!(layer, newfunc)
        t[] += t_step
        sleep(max(0, t_step - (time() - ti)))
    end
    tm = Timer(repeatfunc,0, interval = t_step)
    push!(timers(layer), tm)
    return tm
end
export setBFuncTimer!

function removeTimers!(sim)
    for (idx,tm) in enumerate(timers(sim))
        close(tm)
        deleteat!(timers(sim), idx)
    end

    remB!(graph(sim))
 
end
export removeTimers!


"""
Removes magnetic field
"""
function remB!(layer::IsingLayer)
    bfield(layer) .= Float32(0)

    isnothing(findfirst(x -> x != 0, bfield(graph(layer)))) && setSType!(layer, :Magfield => false)
end

function remB!(g::IsingGraph)
    bfield(g) .= Float32(0)
    setSType!(g, :Magfield => false)
end
export remB!


export plotM
""" 
Plot the magnetic field
"""
function plotM(sim; g = currentLayer(sim))
    imagesc(bfield(g))
end