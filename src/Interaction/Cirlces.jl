# Making circles and offsetting them

# Get a circle centered around i,j  =1, with points ordered from top left to bottom right.
function getOrdCirc(r::Integer)
    if r == 0
        return [(1,1)]
    end

    r2 = r*r
    yoff = -r
    xoff = -1
    lineArray = Vector{Vector{Tuple{Int16, Int16}}}()
    points = 0

    # Looks if point in square at offset mid_off, is within r, if it is, color horizontal line and move down, otherwise, move left and try again.
    while yoff < 1
        # Right lower point
        mid_off =(.5,.5)
        # Midpoint
        # mid_off = (0,0)
        d = (yoff+mid_off[1])^2+(xoff+mid_off[2])^2 #Checks distance to lower left corner from middle
        if d <= r2
            xoff += -1
        else
            xoff += 1
            width = xoff:(-xoff)
            append!(lineArray, [[(1+yoff,1+x) for x in width]])
            yoff += 1
            points += length(width)
        end
    end

    # Append mid line if it's not there
    if lineArray[end][1][1] != 1
        append!(lineArray, [[(1,1+x) for x in (-r):r]])
        points += length((-r):r)
    end
    
    # Mirror all lines, excpept for middle line.
    lines_added = 0
    for offset in 1:(length(lineArray)-1)
        points += length(lineArray[end-offset-lines_added])
        append!(lineArray, [[(1+offset,x) for (y,x) in lineArray[end-offset-lines_added]]])
        lines_added +=1
    end

    # Make new vector with just the points, not the lines and return it
    circPoints = Vector{Tuple{Int16, Int16}}(undef,points)
    p_idx = 1

    for line in lineArray
        for point in line
            circPoints[p_idx] = (point[1],point[2])
            p_idx += 1
        end
    end

    return circPoints

end

# Offset a circle so that center is at i,j
function offCirc(points,i,j)
    circPoints = Vector{Tuple{Int16, Int16}}(undef,length(points))
    p_idx = 1

    for point in points
        circPoints[p_idx] = (point[1]+i-1,point[2]+j-1)
        p_idx += 1
    end

    return circPoints
end

# Remove all points of circle that fall outside of square lattice NxN
function cutCirc(circ,N)
    negPoints = 0 #Points that are out of bounds

    for point in circ
        @inbounds if !(0 < point[1] <= N && 0 < point[2] <= N)
            negPoints += 1
        end
    end

    circPoints = Vector{Tuple{Int16, Int16}}(undef,length(circ) - negPoints)
    p_idx = 1
    for point in circ
        if pointIsOut(point,N)
            circPoints[p_idx] = point
            p_idx +=1
        end
    end
    return circPoints
end

function loopCirc(circ, N)
    circPoints = copy(circ)
    for (pidx, point) in enumerate(circ)
        if pointIsOut(point,N)
            circPoints[pidx] = (latmod(point[1],N),latmod(point[2],N))
        end
    end


    return circPoints
end

function pointIsOut(point::Tuple, N)
    return !(0 < point[1] <= N && 0 < point[2] <= N)
end

pointIsIn(point, N) = !pointIsOut(point, N)

""" Make image from circle, for debugging"""
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