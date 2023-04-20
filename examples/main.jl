# Example File

using InteractiveIsing
using InteractiveIsing.Hamiltonians
using Distributions
using ColorSchemes

# include(joinpath(@__DIR__ , "Learning", "IsingLearning.jl"))
include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))


include(joinpath(@__DIR__,"..","Benchmarking/Benchmarking.jl"))
include(joinpath(@__DIR__,"test.jl"))
#= Radially decreasing weightfunction =#
# weightFunc = radialWF

#= Second nearest neighbor radially falling of weightfunction =#
const weightFunc = isingNN2
# weightFunc = isingNN2
weightFunc.NN = 2
# Test


#= Add randomness to the weights =#
# setAddDist!(weightFunc, Normal(0,0.1))


const sim = IsingSim(
    100,
    50,
    continuous = false, 
    weighted = true;
    weightFunc,
    colorscheme = ColorSchemes.winter
);

const g = sim(true);


# include(joinpath(@__DIR__ , "etest.jl"))
