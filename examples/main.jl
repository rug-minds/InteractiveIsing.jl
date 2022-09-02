# Example File

using InteractiveIsing
using Distributions

include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))
# weightFunc = WeightFunc(
#     (;i,_...) -> sin(i),
#     NN = 1
# )
# setAddDist!(weightFunc, Normal(0,0.1))
# weightFunc = radialWF

const sim = IsingSim(
    continuous = true, 
    graphSize = 512, 
    weighted = false;
    # weightFunc
    )

g = sim(true, async = true);