# User interaction with simulation
"""
Drawing Functions
"""
__precompile__()

module Interaction

include("Cirlces.jl")
# include("../IsingGraphs.jl")
using ..IsingGraphs

export getOrdCirc, circleToState, ordCircleToImg, addRandomDefects!

# Draw a circle to state
function circleToState(g::IsingGraph, circ, i_in,j_in, brush, periodic = false)
    i = Int16(round(i_in))
    j = Int16(round(j_in))
    
    if periodic #Add wrapper to allow for periodic graph
        circle = offCirc(circ,i,j)
    else 
        circle = removeNeg(offCirc(circ,i,j),g.N)
    end

    paintPoints!(g,circle,brush)
    println("Drew circle at y=$i and x=$j")
end

# Make image from circle
function circleToImg(i,j,r, N)
    matr = zeros((N,N))
    circle = getCircle(i,j,r)
    
    for point in circle
        matr[point[1],point[2]] = 1 
    end
    return imagesc(matr)
end

# Make image from circle
function ordCircleToImg(r, N)
    matr = zeros((N,N))
    circle = offCirc(getOrdCirc(r),round(N/2),round(N/2))
    for point in circle
        if point[1] > 0 && point[2] > 0
            if matr[point[1],point[2]] == 1
                matr[point[1],point[2]] = 2
            else 
                matr[point[1],point[2]] = 1 
            end
        end
    end

    return imagesc(matr)
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
