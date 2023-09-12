using InteractiveIsing

simulation = IsingSim(loadGraph())

const g = simulation(false);

createProcess(g)

wg = @WG "(dr, dy, dx) -> (sign(dy)/(abs(dy)+1))" NN = 1
wg2 = @WG "() -> 1" NN = 1

# removeConnections!(g[2])
# genAdj!(g[2], wg2)

# genAdj!(g[1], g[2], wg)

createBaseFig(g, create_singleview)

addLayer!(g, 30,30)
genAdj!(g[6], wg2)