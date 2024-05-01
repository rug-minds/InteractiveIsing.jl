using Preferences, InteractiveIsing
set_preferences!(InteractiveIsing, "precompile_workload" => false; force=true)
# using InteractiveIsing.MCAlgorithms

gr = IsingGraph(precision = Float64, architecture= [(200,200, Continuous),(32,32,Discrete)], sets = [(-1,1),(-1,1)])
wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
genAdj!(gr[1], wg)
simulate(gr)

