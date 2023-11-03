using InteractiveIsing

g = simulate(20,20, continuous = true)
addLayer!(g, 30, 30, type = Discrete)
addLayer!(g, 50, 50, type = Discrete)