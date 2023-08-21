# Example File
using InteractiveIsing, Preferences
using Distributions

const sim = IsingSim(
    200,
    200,
    periodic = true,
    continuous = true, 
    weighted = true,
    # colorscheme = ColorSchemes.winter
);

const g = sim(true);

wg1 = @WeightGenerator "(dr) -> 1/dr" NN = 1
const wg5 = @WeightGenerator "(dr) -> 1/dr" NN = 5

wg10 = @WeightGenerator "(dr) -> 1/dr" NN = 10

addLayer!(g, 100, 100, weightfunc = wg1)

genAdj!(g[1], wg10)
genSPAdj!(g[1], wg10)


# sp_adj(g, tuples2sparse(adj(g)))


# createProcess(sim)

setcoords!(g[1])
setcoords!(g[2], z = 1)
# setcoords!(g[3], z = 2)
# setcoords!(g[4], z = 3)
# setcoords!(g[5], z = 4)

# # # # clampImg!(g, 1, "examples/smileys.jpg")
# connectLayers!(g, 1, 2, (;dr, _...) -> 2, 2)
# connectLayers!(g, 2, 3, (;dr, _...) -> 1, 2)
# connectLayers!(g, 3, 4, (;dr, _...) -> 1, 2)
# connectLayers!(g, 4, 5, (;dr, _...) -> 1, 2)




# addLayer!(sim, 400, 400)
# setcoords!(g[3], z = -1)
# connectLayers!(g, 1, 3, (;dr, _...) -> 1, 2)



# # # overlayNoise!(g, 1, 5, noise_values = [-1,1])``