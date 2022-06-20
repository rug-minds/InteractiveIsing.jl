""" 

Define weighting functions here
Should be a function of the radius r

"""

# Default ising Function
@inline defIsing(r) = r == 1 ? 1. : 0.

function altWeights(r)
    if r % 2 == 1
        return 1.
    elseif r % 2 == 0
        return -1
    else
        return 0
    end
end

function randWeights(r; dist = nothing)
    if r > 1
        return 0.
    end
    
    if dist == nothing
        return rand()
    else
        return rand(dist)
    end
end


