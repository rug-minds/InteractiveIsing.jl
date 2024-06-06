using InteractiveIsing

# Create the graph
ig = IsingGraph(200,200, type = Discrete)
# 
simulate(ig, gui = false, overwrite = true)

# Generate the weights for the IsingModel
wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
genAdj!(ig[1], wg)

# Create a screen to display the graph
layerwindow = LayerWindow(ig[1]);

# Demonstation of the Second Order Phase Transition in the Ising Model
# w = createAnalysisWindow(ig, MT_panel, tstep = 0.01);
# Histersis window
w = createAnalysisWindow(ig, MB_panel);
# Demonstration of Divergence of Isothermal Susceptibility
# w = createAnalysisWindow(ig, χₘ_panel, Tχ_panel, shared_interval = 1/500, tstep = 0.01);


#Anti ferro
# If distance is one, connection is -1
# wg = @WG "dr -> dr == 1 ? -1 : 0" NN=1 
# genAdj!(ig[1], wg)
# simulate(ig, overwrite = true)
