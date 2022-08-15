# User interaction with simulation
"""
Drawing Functions
"""

include("Cirlces.jl")

export getOrdCirc, circleToState, ordCircleToImg, addRandomDefects!, offCirc, loopCirc, reOrderCirc

# Draw a circle to state
function circleToState(g::AbstractIsingGraph, circ, i_in,j_in, brush; periodic = true, clamp = false, imgsize)
    if g.N == imgsize
        i = Int16(round(i_in))
        j = Int16(round(j_in))
    else
        i = Int16(round(i_in/imgsize*g.N))
        j = Int16(round(j_in/imgsize*g.N))
    end        

    offcirc = offCirc(circ,i,j)
    if periodic 
        circle = sort(loopCirc(offcirc,g.N))
    else 
        circle = cutCirc(offcirc,g.N)
    end

    # println("Circle to state circle $circle")

    println("Brush used $brush")
    setSpins!(g,circle,brush,clamp)
    println("Drew circle at y=$i and x=$j")
end

# Add percantage of defects randomly to lattice
function addRandomDefects!(g::IsingGraph,p)
    if length(g.d.aliveList) <= 1 || p == 0
        return nothing
    end

    for _ in 1:round(length(g.d.aliveList)*p/100)
        idx = rand(g.d.aliveList)
        setSpin!(g, idx, Int8(0) , true)
    end

end