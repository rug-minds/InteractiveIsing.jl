# Example File

using InteractiveIsing
using Distributions

include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))
weightFunc = defaultIsingWF
setAddDist!(weightFunc, Normal(0,0.75))

const sim = IsingSim(
    continuous = false, 
    graphSize = 512, 
    weighted = true;
    weightFunc
    )

g = sim(true, async = true);