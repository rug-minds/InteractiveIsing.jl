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
function addRandomDefects!(sim, g, p)
    println("Adding random defects to graph")
    if length(aliveList(g)) <= 1 || p == 0
        return nothing
    end

    al = aliveList(g)
    idxs = al[sample([true, false], Weights([p/100,1-p/100]), length(al))]

    setSpins!(g, idxs, 0, true)

    return

    # for _ in 1:round(length(aliveList(g))*p/100)
    #     setSpins!(sim, g, 0, true)
    # end

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