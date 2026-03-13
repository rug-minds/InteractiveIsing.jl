# Making circles and offsetting them

# Get a circle centered around i,j  =1, with points ordered from top left to bottom right.
function getOrdCirc(r::Integer)
    # println("Making new circ with radius $r")
    circpoints::Vector{Tuple} = []
    r2 = r^2
    for x in (-r):r
        for y in (-r):r
            if x^2+y^2 < r2
                push!(circpoints,tuple(y,x))
            end
        end
    end
    return circpoints
end    
#TODO: Remove export
export getOrdCirc

# Offset a circle so that center is at i,j
function offCirc(points,i,j)
    circPoints = Vector{Tuple{Int, Int}}(undef,length(points))
    p_idx = 1

    for point in points
        circPoints[p_idx] = (point[1]+i-1,point[2]+j-1)
        p_idx += 1
    end

    return circPoints
end

# Remove all points of circle that fall outside of square lattice NxN
function cutCirc(circ, length, width)
    negPoints = 0 #Points that are out of bounds

    for point in circ
        @inbounds if !(0 < point[1] <= length && 0 < point[2] <= width)
            negPoints += 1
        end
    end

    circPoints = Vector{Tuple{Int, Int}}(undef,length(circ) - negPoints)
    p_idx = 1
    for point in circ
        if pointIsOut(point, length, width)
            circPoints[p_idx] = point
            p_idx +=1
        end
    end
    return circPoints
end

function loopCirc(circ, length, width)
    circPoints = copy(circ)
    for (pidx, point) in enumerate(circ)
        if pointIsOut(point, length, width)
            circPoints[pidx] = (latmod(point[1],Int(length)),latmod(point[2],Int(width)))
        end
    end
    return circPoints
end

@inline function pointIsOut(point::Tuple, length, width)
    return !(0 < point[1] <= length && 0 < point[2] <= width)
end

pointIsIn(point, length, width) = !pointIsOut(point, length, width)

""" Make image from circle, for debugging"""
# Make image from circle
function ordCircleToImg(r)
    mat = zeros(2*r-1,2*r-1)
    circ = getOrdCirc(r)
    for point in circ
        mat[point[1] + r ,point[2] + r ] = 1
    end
    imagesc(mat)
end

function ordCircleToMat(r)
    mat = zeros(2*r-1,2*r-1)
    circ = getOrdCirc(r)
    for point in circ
        mat[point[1] + r ,point[2] + r ] = 1
    end
    return mat
end
export ordCircleToMat
