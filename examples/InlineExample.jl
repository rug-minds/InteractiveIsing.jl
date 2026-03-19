using InteractiveIsing
using InteractiveIsing.Processes

g = IsingGraph(100,100,
                Discrete(),
                (@WG (;dr) -> dr == 1 ? 1 : 0 NN=1) ) 

comp = g.default_algorithm
p = InlineProcess(comp, Input(comp; state = g), lifetime = 10)