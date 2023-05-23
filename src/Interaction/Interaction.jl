# User interaction with simulation
#=
Drawing Functions
=#

include("Cirlces.jl")

export getOrdCirc, circleToState, ordCircleToImg, addRandomDefects!, offCirc, loopCirc, reOrderCirc

# Draw a circle to state
function circleToState(sim, g, i_in, j_in, brush; periodic = true, clamp = false, imgsize, debug = false)

    # If img is size of underlying graph, use rounded coordinate, otherwise scale
    i = glength(g) == imgsize[1]    ? Int16(round(i_in)) : Int16(round(i_in/imgsize*glength(g)))
    j = gwidth(g) == imgsize[2]     ? Int16(round(j_in)) : Int16(round(j_in/imgsize*gwidth(g)))
   
    offcirc = offCirc(circ(sim),i,j)

    if periodic 
        circle = sort(loopCirc(offcirc, glength(g), gwidth(g)), by = x -> reverse(x))
    else 
        circle = cutCirc(offcirc, glength(g), gwidth(g))
    end

    if debug
        println("i_in $i_in, j_in $j_in")
        println("i $i, j$j")
        println("Brush used $brush")
        # println("Drew circle at y=$i and x=$j")
        println("Circle to state circle $cir|cle")
    end

    setSpins!(g, circle, brush, clamp)
    
end

# Add percantage of defects randomly to lattice
function addRandomDefects!(layer, p)
    # if length(aliveList(layer)) <= 1 || p == 0
    #     return nothing
    # end

    println("Adding random defects to layer")
    # al = aliveList(g)
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


# function addRandomDefects!(sim, layer::IsingLayer, p)
#     println("Adding random defects to layer")
#     if length(aliveList(layer)) <= 1 || p == 0
#         return nothing
#     end
#     al = aliveList(layer)
#     idxs = al[sample([true, false], Weights([p/100,1-p/100]), length(al))]

#     setSpins!(sim, g, idxs, 0, true)

#     # for _ in 1:round((length(state(layer))-ndefects(layer) )*p/100)
#     #     setSpins!(sim, layer, Int32(rand(aliveList(layer))), Int32(0), true)
#     # end

# end