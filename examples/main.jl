# Example File
using InteractiveIsing, Preferences
set_preferences!(InteractiveIsing, "precompile_workload" => true; force=true)
using Distributions
# using ColorSchemes

# include(joinpath(@__DIR__ , "Learning", "IsingLearning.jl"))
# include(joinpath(@__DIR__ , "WeightFuncsCustom.jl"))


# include(joinpath(@__DIR__,"..","Benchmarking/Benchmarking.jl"))
# include(joinpath(@__DIR__,"test.jl"))
#= Radially decreasing weightfunction =#
# weightFunc = radialWF

#= Second nearest neighbor radially falling of weightfunction =#
# const weightFunc = isingNN2
# weightFunc = isingNN2
# weightFunc.NN = 1

#= Add randomness to the weights =#
# setAddDist!(weightFunc, Normal(0,0.1))

const sim = IsingSim(
    500,
    500,
    periodic = true,
    continuous = true, 
    weighted = true,
    # colorscheme = ColorSchemes.winter
);

const g = sim(true);

# # Add Layers
# addLayer!(g, 350, 350)


wg = @WeightGenerator "(dr) -> dr == 1" NN = 1

genAdj!(g[1], wg)

# createProcess(sim)

# setcoords!(g[1])
# setcoords!(g[2], z = 1)

# # # clampImg!(g, 1, "examples/smileys.jpg")
# connectLayers!(g, 1, 2, (;dr, _...) -> 1, 2)

# addLayer!(sim, 400, 400)
# setcoords!(g[3], z = -1)
# connectLayers!(g, 1, 3, (;dr, _...) -> 1, 2)



# # # overlayNoise!(g, 1, 5, noise_values = [-1,1])