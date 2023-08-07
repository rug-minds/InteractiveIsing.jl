export drawCircle, addRandomDefects!, setBField!, remBField!, setClamp!, remClamp!
"""
Draw circle of some size to the layer g with center at i,j, and state value of val.
"""
function drawCircle(layer, i_in, j_in, val, r = nothing; periodic = true, clamp = false, debug = false)
    fsim = sim(graph(layer))

    imgsize = size(image(fsim)[])
    
    # If img is size of underlying graph, use rounded coordinate, otherwise scale
    i = glength(layer) == imgsize[1]    ? Int16(round(i_in)) : Int16(round(i_in/imgsize*glength(layer)))
    j = gwidth(layer) == imgsize[2]     ? Int16(round(j_in)) : Int16(round(j_in/imgsize*gwidth(layer)))
   
    circle = isnothing(r) ? circ(fsim) : getOrdCirc(r)


    offcirc = offCirc(circle, i,j)

    if periodic 
        circle = sort(loopCirc(offcirc, glength(layer), gwidth(layer)), by = x -> reverse(x))
    else 
        circle = cutCirc(offcirc, glength(layer), gwidth(layer))
    end

    if debug
        println("i_in $i_in, j_in $j_in")
        println("i $i, j$j")
        println("Value used $val")
        # println("Drew circle at y=$i and x=$j")
        println("Circle to state circle $cir|cle")
    end

    setSpins!(layer, circle, val, clamp)
    
end

"""
Randomly make p percent of states in layer defect
"""
function addRandomDefects!(layer, p)
    println("Adding random defects to layer")

    nonzeros = Int32[]
    for i in 1:length(state(layer))
        if nand(state(layer)[i] == 0, isDefect(graph(layer))[idxLToG(layer, i)] )
            push!(nonzeros, i)
        end
    end

    # If is empty do nothing
    isempty(nonzeros) && return nothing

    # From nonzeros or non-defects choose p% to set to 0 and defect
    idxs = nonzeros[sample([true, false], Weights([p/100,1-p/100]), length(nonzeros))]

    setSpins!(layer, idxs, 0, true)

    return
end

"""
Set a magneticfield to a layer
"""

"""
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
        setBFuncTimer!(layer, (;x,y) -> func(x,y))
    else
        error("Mode $mode not recognized")
    end
end
"""
Set magnetic field by idxs and strength. 
"""
setBField!(layer::AbstractIsingLayer, idxs::Vector, strengths::Vector) = setBIdxs!(layer, idxs, strengths)
#remBField!

"""
Add a term to the Hamiltonian of the form beta 1/2(sigma_i - y)^2 where sigma_i is the i-th state and y is the target for that state
Give a function in the form of (x,y)
"""
setClamp!(layer::AbstractIsingLayer, func::Function) = setClampFunc!(layer, (;x,y) -> func(x,y))
setClamp!(layer::AbstractIsingLayer, idxs::Vector, strengths::Vector) = setClampIdxs!(layer, idxs, strengths)
#remClamp!
