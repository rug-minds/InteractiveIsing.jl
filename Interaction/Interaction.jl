# User interaction with simulation
"""
Drawing Functions
"""
__precompile__()

module Interaction

include("Cirlces.jl")
using ..IsingGraphs

export getOrdCirc, circleToState, ordCircleToImg, addRandomDefects!

# Put a lattice index (i or j) back onto lattice by looping it around
@inline function latmod(i,N)
    if i < 1 || i > N
        return i = mod((i-1),N) +1
    end
    return i
end

# Draw a circle to state
function circleToState(g::IsingGraph, circ, i_in,j_in, brush, periodic = true)
    i = Int16(round(i_in))
    j = Int16(round(j_in))
    
    if periodic #Add wrapper to allow for periodic graph
        circle = loopCirc(offCirc(circ,i,j),g.N)
    else 
        circle = cutCirc(offCirc(circ,i,j),g.N)
    end

    paintPoints!(g,circle,brush)
    println("Drew circle at y=$i and x=$j")
end

# Add percantage of defects randomly to lattice
function addRandomDefects!(g::IsingGraph,p)
    if length(g.aliveList) <= 1 || p == 0
        return nothing
    end

    for def in 1:round(length(g.aliveList)*p/100)
        idx = rand(g.aliveList)
        addDefect!(g,idx)
    end

end

end
