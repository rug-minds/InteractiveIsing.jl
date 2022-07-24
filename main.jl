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

qmlfile = joinpath(dirname(Base.source_path()), "qml", "Main.qml")

using QML
using Images

include("WeightFuncsCustom.jl")

const sim = Sim(
    continuous = false, 
    graphSize = 500, 
    weighted = false, 
    weightFunc = defaultIsingWF
    )

sim(true)
