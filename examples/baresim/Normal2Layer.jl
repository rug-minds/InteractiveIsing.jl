using InteractiveIsing
g = IsingGraph(Float32, architecture = [(500,500),(250,250)], sets = [(1,1)], [(1,1)])
loadparams(g, "examples/baresim/2layer.jld2")
simulate(g)