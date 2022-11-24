# Example File

using InteractiveIsing
using InteractiveIsing.Hamiltonians
using Distributions

# include(joinpath(@__DIR__ , "Learning", "IsingLearning.jl"))
include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))

include(joinpath(@__DIR__,"..","Benchmarking/Benchmarking.jl"))
include(joinpath(@__DIR__,"test.jl"))
#= Radially decreasing weightfunction =#
# weightFunc = radialWF

#= Second nearest neighbor radially falling of weightfunction =#
weightFunc = isingNN2

#= Add randomness to the weights =#
# setAddDist!(weightFunc, Normal(0,0.1))


const sim = IsingSim(
    continuous = false, 
    graphSize = 500, 
    weighted = true;
    weightFunc
)

g = sim(true);