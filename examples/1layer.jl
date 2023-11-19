using InteractiveIsing
g = IsingGraph(400,400, type = Discrete)

simulate(g, overwrite = true)

wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
genAdj!(g[1], wg)

midpanel(getml())["image"].colorrange[] = (-1,1)

# wg_a = @WG "dr -> dr == 1 ? -1 : 0" NN=1
# genAdj!(g[1], wg_a)

using LsqFit
@. model(x, p) = p[1] * exp(-x/p[2])
# w = createAnalysisWindow(g, MT_panel, tstep = 0.01)
w = createAnalysisWindow(g, MB_panel, tstep = 0.01)
# function getcorr()
#     ydata = w["corr"][]
#     xdata = w["corr_r"][]
#     return xdata, ydata
# end

# xdata, ydata = getcorr()
# fit = curve_fit(model, xdata, ydata, [1.0, 1.0])
# fit.param