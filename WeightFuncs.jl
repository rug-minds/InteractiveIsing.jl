""" 

Define weighting functions here
Should be a function of the radius dr

"""
__precompile__()

module WeightFuncs

using Distributions

export WeightFunc, setAddDist, setMultDist, DefaultIsing, AltWeights, AltWeightsCross, RandWeights

# Assuming one method,
# Returns symbols of argnames
function func_argnames(f::Function)
    ml = collect(methods(f))
    return Base.method_argnames(last(ml))[2:end]
end


mutable struct WeightFunc
    f::Function
    NN::Integer
    periodic::Bool
    invoke::Function
    addTrue::Bool
    multTrue::Bool
    addDist::Any 
    multDist::Any

    function WeightFunc(func::Function, ; NN::Integer)
        invoke(dr,i,j) = func(;dr,i,j)
        return new(func,Int8(NN), true, invoke, false, false)
    end

end


# Add an additive distribution
function setAddDist(weightfunc, dist)
    weightfunc.addTrue = true
    weightfunc.addDist = dist

    invoke(dr,i,j) = !weightfunc.addTrue ? rand(weightfunc.addDist) + weightfunc.f(;dr,i,j) : rand(weightfunc.multDist)*weightfunc.f(;dr,i,j)+rand(weightfunc.addDist)
    
    weightfunc.invoke = invoke
end

# Add an additive distribution
function setMultDist(weightfunc, dist)
    weightfunc.multTrue = true
    weightfunc.multDist = dist

    invoke(dr,i,j) = !weightfunc.addTrue ? rand(weightfunc.multDist)*weightfunc.f(;dr,i,j) : rand(weightfunc.multDist)*weightfunc.f(;dr,i,j)+rand(weightfunc.addDist)

    weightfunc.invoke = invoke
end

# Default ising Function
DefaultIsing =  WeightFunc(
    (;dr, _...) -> dr == 1 ? 1. : 0., 
    NN = 1
)

altWeights = WeightFunc(
    (;dr, _...) ->
        if dr % 2 == 1
            return 1.0*1/dr^2
        elseif dr % 2 == 0
            return -1*1/dr^2
        else
            return 1/dr^2
        end,
    NN  = 2
)

AltWeightsCross = WeightFunc(
    (;dr, _...) ->
        if dr % 2 == 1
            return -1
        elseif dr % 2 == 0
            return 1
        else
            return 0
        end,
    NN = 2
)

RandWeights = WeightFunc(
    (;dr, _...) ->
        if dist == nothing
            return rand()
        else
            return rand(dist)
        end,
    NN = 1
)


""" Old """
# Use distribution centered around zero
function randomizeWeights(dr, func , dist)::Float32
    weight = func(dr)
    if weight == 0
        return 0.
    else
        return weight + rand(dist)
    end
end

end


