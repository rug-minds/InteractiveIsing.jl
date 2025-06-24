export drawCircle, addRandomDefects!, setBField!, remBField!, setClamp!, remClamp!
"""
Draw circle of some size to the layer g with center at i,j, and state value of val.
"""
function drawCircle(layer, x_in, y_in, val, r = nothing; periodic = true, clamp = false, debug = false)
    fsim = sim(graph(layer))

    if !( 1 <= x_in <= size(layer)[1] && 1 <= y_in <= size(layer)[2])
        return
    end

    # imgsize = size(image(fsim)[])
    
    # If img is size of underlying graph, use rounded coordinate, otherwise scale
    # i = glength(layer) == imgsize[1]    ? Int16(round(x_in)) : Int16(round(x_in/imgsize[1]*glength(layer)))
    # j = gwidth(layer) == imgsize[2]     ? Int16(round(y_in)) : Int16(round(y_in/imgsize[2]*gwidth(layer)))
    i = round(Int, x_in)
    j = round(Int, y_in)

    circle = isnothing(r) ? circ(fsim) : getOrdCirc(r)


    offcirc = offCirc(circle, i,j)

    if periodic 
        circle = sort(loopCirc(offcirc, glength(layer), gwidth(layer)), by = x -> reverse(x))
    else 
        circle = cutCirc(offcirc, glength(layer), gwidth(layer))
    end

    if debug
        println("x_in $x_in, y_in $y_in")
        println("i $i, j$j")
        println("Value used $val")
        # println("Drew circle at y=$i and x=$j")
        println("Circle to state circle $cir|cle")
    end

    setSpins!(layer, circle, val, clamp)
    return
end


# TODO: Notify sim
"""
Randomly make p percent of states in layer defect
"""
function addRandomDefects!(layer, p, val = 0)
    println("Adding random defects to layer")

    nonzeros = Int32[]
    for i in 1:length(state(layer))
        if nand(state(layer)[i] == 0, isDefect(graph(layer))[idxLToG(i, layer)] )
            push!(nonzeros, i)
        end
    end

    # If is empty do nothing
    isempty(nonzeros) && return nothing

    # From nonzeros or non-defects choose p% to set to 0 and defect
    idxs = nonzeros[sample([true, false], Weights([p/100,1-p/100]), length(nonzeros))]

    setSpins!(layer, idxs, val, true)

    refresh(graph(layer))

    return
end

"""
Set a magneticfield to a layer

Enter a funcion f(x,y)
Can use mode :Static, :Timed, :Repeating, :Timer

For repeating you can supply a timestep as final argument
For you timed can supply a time length and time step as final arguments
For you timer can supply a time step as final argument

"""
function setBField!(layer::AbstractIsingLayer, func::Function, mode::Symbol = :Static, args...)
    if mode == :Static
        setBFunc!(layer, (;x,y) -> func(x,y))
    elseif mode == :Repeating
        setBFuncRepeating!(layer, (;x,y) -> func(x,y))
    elseif mode == :Timed
        setBFuncTimed!(layer, (;x,y) -> func(x,y), args...)
    elseif mode == :Timer
        setBFuncTimer!(layer, (;x,y,t) -> func(x,y,t))
    else
        error("Mode $mode not recognized")
    end
end
"""
Set magnetic field by idxs and strength. 
"""
setBField!(layer::AbstractIsingLayer, idxs::Vector, strengths::Vector) = setBIdxs!(layer, idxs, strengths)
export setBField!
#remBField!

"""
Add a term to the Hamiltonian of the form beta 1/2(sigma_i - y)^2 where sigma_i is the i-th state and y is the target for that state
Give a function in the form of (x,y)
"""
setClamp!(layer::AbstractIsingLayer, func::Function) = setClampFunc!(layer, (;x,y) -> func(x,y))
setClamp!(layer::AbstractIsingLayer, idxs::Vector, strengths::Vector) = setClampIdxs!(layer, idxs, strengths)
function setClamp!(l::AbstractIsingLayer, idx::Integer, strength::Real)
    defects(l)[idx] = true
    state(l)[idx] = 1
    return state(l)
end
function setClamp!(l::AbstractIsingLayer, strength::Real)
    setdefectrange!(l, Int32[1:length(state(l));])
    state(l) .= strength
    return state(l)
end
#remClamp!

function globalB!(g, strength)
    bfield(g) .= strength
end

function setB!(g::IsingGraph, val::Real, idxs::AbstractVector)
    bfield(g)[idxs] .= val
end

setB!(l::IsingLayer, val) = setB!(graph(l), val, graphidxs(l))
setB!(val::Real, ls::IsingLayer...) = for l in ls; setB!(l, val); end
setB!(l::IsingLayer, strengths::AbstractVector) = setBIdxs!(graph(l), (@view graphidxs(l)[1:length(strengths)]), strengths)
export globalB!, setB!

avgB(l::IsingLayer) = sum(bfield(l))/length(bfield(l))
export avgB