using InteractiveIsing

g = simulate(500,500, continuous = true)

w_generator = @WG "(dr) -> 1/dr" NN = 1

genAdj!(g[1], w_generator)

addLayer!(g, 400, 400, w_generator)

layer_w_generator = @WG "(dr) -> dr" NN = 2


genAdj!(g[1], g[2], layer_w_generator)

setBFuncTimer!(g[1], (;x,y,t)-> 2*sin(pi*x/10 - pi*t))
setBFuncTimer!(g[2], (;x,y,t)-> 2*sin(pi*y/10 - pi*t))
remB!(g)


removeLayer!(g,2)

circ_layer_gen = @WG "(dr, x,y) -> let dm = sqrt((x-200)^2+(y-200)^2); 1/dr*(1-dm/(sqrt(2*200^2))) end" NN = 2
genAdj!(g[2], w_generator)
genAdj!(g[1],g[2], w_generator)
removeConnections!(g[2])