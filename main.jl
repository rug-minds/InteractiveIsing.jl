push!(LOAD_PATH, pwd())
push!(LOAD_PATH, pwd()*"/Interaction")
push!(LOAD_PATH, pwd()*"/Learning")


# include("IsingSim.jl")
using IsingSim
using GPlotting
using IsingGraphs
using Interaction
using IsingMetropolis
using WeightFuncs
using Analysis


include("WeightFuncsCustom.jl")

const sim = Sim(
    continuous = false, 
    graphSize = 512, 
    weighted = false, 
    weightFunc = defaultIsingWF
    )

sim(true)