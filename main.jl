push!(LOAD_PATH, pwd())
push!(LOAD_PATH, pwd()*"/Interaction")
push!(LOAD_PATH, pwd()*"/Learning")


using IsingSim
using GPlotting
using IsingGraphs
using Interaction
using IsingMetropolis
using WeightFuncs
using Analysis
using Distributions

include("WeightFuncsCustom.jl")

weightFunc = defaultIsingWF
setAddDist!(weightFunc, Normal(0,0.3))

const sim = Sim(
    continuous = false, 
    graphSize = 512, 
    weighted = true;
    weightFunc
    )

g = sim(true)
