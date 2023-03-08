#=

Define weighting functions here
Should be a function of the radius dr

=#

export WeightFunc, setAddDist!, setMultDist!, defaultIsingWF, altRWF, altCrossWF, randWF

# Assuming one method,
# Returns symbols of argnames
function func_argnames(f::Function)
    ml = collect(methods(f))
    return Base.method_argnames(last(ml))[2:end]
end

struct WeightType{Add,Mult,Periodic,Self} end

mutable struct WeightFunc
    NN::Integer
    selfweight::Function
    func::Function
    wt::WeightType
    addDist::Any 
    multDist::Any
end

WeightFunc(func; NN = 1, periodic = true, self = false, selfweight = (i,j) -> 1) = WeightFunc(NN, selfweight, func, WeightType{false,false,periodic,self}(), (;_...) -> 1, (;_...) -> 1)

@setterGetter WeightFunc
periodic(wt::WeightType{A,B,Periodic,C}) where {A,B,Periodic,C} = Periodic
periodic(wf::WeightFunc) = periodic(wt(wf))
self(wt::WeightType{A,B,C,Self}) where {A,B,C,Self} = Self
self(wf::WeightFunc) = self(wt(wf))
self(wf::WeightFunc, selfval; weighttype::WeightType{A,B,C,D} = wt(wf)) where {A,B,C,D} = wf.wt = WeightType{A,B,C,selfval}()
export periodic, self




# Add an additive distribution
function setAddDist!(wf, dist, func= (;dr,i,j,_...) -> 1)
    addDist(wf, (;dr,i,j,args...) -> func(;dr,i,j,args...)*dist)
    wtparams = typeof(wt(wf)).parameters
    wt(wf, WeightType{true, wtparams[2]}())
end

function setMultDist!(wf, dist, func= (;dr,i,j,_...) -> 1)
    multDist(wf, (;dr,i,j,args...) -> func(;dr,i,j,args...)*dist)
    wtparams = typeof(wt(wf)).parameters
    wt(wf, WeightType{wtparams[1], true}())
end

import Base: *
*(s::String, b::Bool) = b ? s : ""
*(b::Bool, s::String) = *(s,b)

@generated function getWeight(wt::WeightType{Add,Mult}, func, addDist, multDist, dr, i ,j) where {Add,Mult}
    addString = "+rand(addDist(;dr,i,j))"
    multString = "rand(multDist(;dr,i,j))*"
    generalString = "func(;dr,i,j)"
    finalstring = (Mult*multString)*generalString*(Add*addString)
    return Meta.parse(finalstring)
end

getWeight(wf::WeightFunc, dr, i ,j) = getWeight(wt(wf), func(wf), addDist(wf), multDist(wf), dr, i ,j)
export getWeight

# mutable struct WeightFunc
#     f::Function
#     NN::Integer
#     periodic::Bool
#     self::Bool
#     invoke::Function
#     addTrue::Bool
#     multTrue::Bool
#     addDist::Any 
#     multDist::Any

#     function WeightFunc(func::Function, ; NN::Integer)
#         invoke(dr,i,j) = func(;dr,i,j)
#         return new(func, Int32(NN), true, false, invoke, false, false)
#     end

# end


# # Add an additive distribution
# function setAddDist!(weightFunc, dist, func= (;dr,i,j,_...) -> 1)
#     weightFunc.addTrue = true
#     weightFunc.addDist = (;dr,i,j) -> func(;dr,i,j)*dist

    
#     if !weightFunc.multTrue
#         inv = (dr,i,j) -> rand(weightFunc.addDist(;dr,i,j)) + weightFunc.f(;dr,i,j)
#     else
#         inv = (dr,i,j) -> rand(weightFunc.multDist(;dr,i,j))*weightFunc.f(;dr,i,j)+rand(weightFunc.addDist(;dr,i,j))
#     end

#     weightFunc.invoke = inv
    
# end

# # Add an additive distribution
# function setMultDist!(weightFunc, dist, func= (;dr,i,j,_...) -> 1)
#     weightFunc.multTrue = true
#     weightFunc.multDist = (;dr,i,j) -> func(;dr,i,j)*dist

#     if !weightFunc.addTrue
#         inv = (dr,i,j) -> rand(weightFunc.multDist(;dr,i,j))*weightFunc.f(;dr,i,j)
#     else
#         inv = (dr,i,j) -> rand(weightFunc.multDist(;dr,i,j))*weightFunc.f(;dr,i,j)+rand(weightFunc.addDist(;dr,i,j))
#     end

#     weightFunc.invoke = inv
# end

# Default ising Function
defaultIsingWF =  WeightFunc(
    (;dr, _...) -> dr == 1 ? 1. : 0., 
    NN = 1
)

altRWF = WeightFunc(
    (;dr, _...) ->
        if dr % 2 == 1
            return -1.0*1/dr^2
        elseif dr % 2 == 0
            return 1*1/dr^2
        else
            return -1/dr^2
        end,
    NN  = 2
)

altCrossWF = WeightFunc(
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

randWF = WeightFunc(
    (;dr, _...) ->
        if dist == nothing
            return rand()
        else
            return rand(dist)
        end,
    NN = 1
)


# """ Old """
# Use distribution centered around zero
function randomizeWeights(dr, func , dist)::Float32
    weight = func(dr)
    if weight == 0
        return 0.
    else
        return weight + rand(dist)
    end
end