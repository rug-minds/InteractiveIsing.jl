""" 

Define weighting functions here
Should be a function of the radius r

"""

# Default ising Function
@inline defaultIsing(r)::Float32 = r == 1 ? 1. : 0.

function altWeightsCross(r)::Float32
    if r % 2 == 1
        return 1.0*1/r^2
    elseif r % 2 == 0
        return -1*1/r^2
    else
        return 1/r^2
    end
end

function altWeights(r)::Float32
    if r % 2 == 1
        return 1
    elseif r % 2 == 0
        return -1
    else
        return 0
    end
end

function randWeights(r; dist = nothing)::Float32
    if r > 1
        return 0.
    end

    if dist == nothing
        return rand()
    else
        return rand(dist)
    end
end

# Use distribution centered around zero
function randomizeWeights(weight::Float32, dist)::Float32
    if weight == 0
        return 0.
    else
        return weight + rand(dist)
    end
end



