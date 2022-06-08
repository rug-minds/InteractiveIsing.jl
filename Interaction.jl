# Special routines for program
"""
Drawing Functions
"""
# Get a circle centered around i,j  =1
function getOrdCirc(r)
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

# Center circle at i,j
function offCirc(points,i,j, periodic = false)
    circPoints = Vector{Tuple{Int16, Int16}}(undef,length(points))
    p_idx = 1

    for point in points
        circPoints[p_idx] = (point[1]+i-1,point[2]+j-1)
        p_idx += 1
    end

    return circPoints
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


# Draw a circle to state
function circleToState(g, circ, i,j, brush, periodic = false)
    println("Drew circle at y=$i and x=$j")
    
    if periodic
        circle = offCirc(circ,i,j)
    else
        circle = removeNeg(offCirc(circ,i,j),g.N)
    end

    paintPoints!(g,circle,brush)
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


#Removing 
function removeNeg(circ,N)
    negPoints = 0

    for point in circ
        @inbounds if !(0 < point[1] < N && 0 < point[2] < N)
            negPoints += 1
        end
    end

    circPoints = Vector{Tuple{Int16, Int16}}(undef,length(circ) - negPoints)
    p_idx = 1
    for point in circ
        if 0 < point[1] < N && 0 < point[2] < N
            circPoints[p_idx] = point
            p_idx +=1
        end
    end
    return circPoints
end



#OLD
function getCircle(i,j,r)
    i = round(Int,i)
    j = round(Int,j)

    # if radius = 0, return point
    if r == 0
        return [(i,j)]
    end

    r2 = r*r
    lineArray = [(i,y) for y in (j-r):(j+r)] #List of Coordinates, starts with vertical line centered around i,j, with radius r
    xoff = 1
    yoff = r
    # Tracks how often algorithm goes down
    steps = 0
    while xoff != yoff
        # Left lower point
        mid_off =(-.5,-.5)
        # Midpoint
        # mid_off = (0,0)
        d = (xoff+mid_off[1])^2+(yoff+mid_off[2])^2 #Checks distance to lower left corner from middle
        if d < r2
            # The line in question
            append!(lineArray, [(i+xoff,y) for y in (j-yoff):(j+yoff)])
            # The line reflected in y axis
            append!(lineArray, [(i-xoff,y) for y in (j-yoff):(j+yoff)])

            xoff +=1 #If succesful, move square to right
        else
            # If moving down, then line reflected in diagonal is added
            append!(lineArray,[(i+r-steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
            # Also do it for negative side
            append!(lineArray,[(i-r+steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
            yoff -= 1 # If unsuccesful, move down
            steps+=1
        end
    end
    # Add last two lines reflected in diagonal
    append!(lineArray,[(i+r-steps,y) for y in (j-(xoff)):(j+(xoff))])
    append!(lineArray,[(i-r+steps,y) for y in (j-(xoff)):(j+(xoff))])
    # append!(lineArray,[(i+r-steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
    # append!(lineArray,[(i-r+steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
    return lineArray
end