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
wg5 = @WeightGenerator "(dr) -> 1/dr" NN = 5

wgt = @WeightGenerator wg1 weightFunc = "(dr) -> 1/dr" NN = 1

wg10 = @WeightGenerator "(dr) -> 1/dr" NN = 10

# for _ in 1:4; addLayer!(g, 100, 100, weightfunc = wg1); end

addLayer!(g, 100, 100, weightfunc = wg1)
addLayer!(g, 100, 100, weightfunc = wg5)
addLayer!(g, 75, 75, weightfunc = wg1)
addLayer!(g, 100, 100, weightfunc = wg1)

genSPAdj!(g[1], wg10)


setcoords!(g[1])
setcoords!(g[2], z = 1)
setcoords!(g[3], z = 2)
setcoords!(g[4], z = 3)
setcoords!(g[5], z = 4)

genSPAdj!(g[1],g[2], wg5)
genSPAdj!(g[2],g[3], wg1)
genSPAdj!(g[3],g[4], wg1)
genSPAdj!(g[4],g[5], wg1)



# sp_adj(g, tuples2sparse(adj(g)))


# # createProcess(sim)



# genSPAdj!(g[1], g[2], wg10)


# addLayer!(sim, 400, 400)
# setcoords!(g[3], z = -1)
# connectLayers!(g, 1, 3, (;dr, _...) -> 1, 2)



# # # overlayNoise!(g, 1, 5, noise_values = [-1,1])