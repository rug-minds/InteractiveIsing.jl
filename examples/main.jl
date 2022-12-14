# Example File

using InteractiveIsing
using InteractiveIsing.Hamiltonians
using Distributions
# using Revise

# include(joinpath(@__DIR__ , "Learning", "IsingLearning.jl"))
include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))

include(joinpath(@__DIR__,"..","Benchmarking/Benchmarking.jl"))
include(joinpath(@__DIR__,"test.jl"))
#= Radially decreasing weightfunction =#
# weightFunc = radialWF

#= Second nearest neighbor radially falling of weightfunction =#
# weightFunc = isingNN2
weightFunc = defaultIsingWF

#= Add randomness to the weights =#
# setAddDist!(weightFunc, Normal(0,0.1))


const sim = IsingSim(
    continuous = false, 
    length = 300,
    width = 500, 
    weighted = false;
    weightFunc
)

const g = sim(true);