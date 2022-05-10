# Special routines for program

function paintPoint!(g,i,j, brush)
    idx = coordToIdx(i,j,g.N)

    if brush == 0
        addDefect!(g,idx)
    else
        setEl!(g,idx,brush)
    end
end

# Restores all defects to random states
function restoreDefects!(g)
    nDefects = length(g.defectList)
    states = initRandomState(nDefects)
    g.state[g.defectList] .= states
    append!(g.aliveList,g.defectList)
end




"""
Drawing Functions
"""
# Get a circle of states around lattice point i,j
function getCircle(i,j,r)
    i = round(Int,i)
    j = round(Int,j)

    # if radius = 0, return point
    if r == 0
        return [(i,j)]
    end

    r2 = r*r
    coordl = [(i,y) for y in (j-r):(j+r)] #List of Coordinates, starts with vertical line centered around i,j, with radius r
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
            append!(coordl, [(i+xoff,y) for y in (j-yoff):(j+yoff)])
            # The line reflected in y axis
            append!(coordl, [(i-xoff,y) for y in (j-yoff):(j+yoff)])

            xoff +=1 #If succesful, move square to right
        else
            # If moving down, then line reflected in diagonal is added
            append!(coordl,[(i+r-steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
            # Also do it for negative side
            append!(coordl,[(i-r+steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
            yoff -= 1 # If unsuccesful, move down
            steps+=1
        end
    end
    # Add last two lines reflected in diagonal
    append!(coordl,[(i+r-steps,y) for y in (j-(xoff)):(j+(xoff))])
    append!(coordl,[(i-r+steps,y) for y in (j-(xoff)):(j+(xoff))])
    # append!(coordl,[(i+r-steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
    # append!(coordl,[(i-r+steps,y) for y in (j-(xoff-1)):(j+(xoff-1))])
    return coordl
end

# Draw a circle to state
function circleToState(g,i,j,r,brush)
    println("Drew circle at y=$i and x=$j")

    circle = getCircle(i,j,r)
    
    for point in circle
        # if point falls out of lattice, continue
        if point[1] < 1 || point[2] <1 || point[1] > g.N || point[2] > g.N
            continue
        end
        paintPoint!(g,point[1],point[2],brush)
    end

end

# Make image from circle
function circleToImg(i,j,r, N)
    matr = zeros((N,N))
    circle = getCircle(i,j,r)
    
    for point in circle
        # println("Point $point")
        matr[point[1],point[2]] = 1 
    end
    return imagesc(matr)
end