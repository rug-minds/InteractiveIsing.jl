# Example File

using InteractiveIsing
using Distributions
# include(joinpath(@__DIR__ , "Learning", "IsingLearning.jl"))
include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))

# Radially decreasing weightfunction
# weightFunc = radialWF

# Second nearest neighbor radially falling of weightfunction
weightFunc = isingNN2

# Add randomness to the weights
# setAddDist!(weightFunc, Normal(0,0.1))


const sim = IsingSim(
    continuous = true, 
    graphSize = 500, 
    weighted = true;
    weightFunc
    )

g = sim(true);