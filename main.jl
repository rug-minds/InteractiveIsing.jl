push!(LOAD_PATH, pwd())
push!(LOAD_PATH, pwd()*"/Interaction")
push!(LOAD_PATH, pwd()*"/Learning")

# using QML
using IsingSim
using Distributions

include("WeightFuncsCustom.jl")

weightFunc = defaultIsingWF
# setAddDist!(weightFunc, Normal(0,0.5))

const sim = Sim(
    continuous = true, 
    graphSize = 100, 
    weighted = true;
    weightFunc
    )

g = sim(true);

