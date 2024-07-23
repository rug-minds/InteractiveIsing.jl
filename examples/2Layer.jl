using InteractiveIsing, BenchmarkTools
import InteractiveIsing as II

wg = @WG "(dr) -> 1/dr" NN = 5

layer_connections = @WG "(dr, dx, dy) -> 1" NN = 3

g = IsingGraph(500, 500)

# @benchmark II.genLayerConnections(g[1],wg)
# @benchmark II.genLayerConnectionsOLD(g[1],wg)
# @benchmark II.genLayerConnectionsNEW(g[1],wg)
