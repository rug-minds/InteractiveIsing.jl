using InteractiveIsing

g = simulate(28,28, type = Continuous, set = (0f0, 1f0))
addLayer!(g, 10,50, type = Discrete, set = (0f0, 1f0))
addLayer!(g, 2,10, type = Discrete, set = (0f0, 1f0))

using DelimitedFiles
v_bias = readdlm("examples/bin/train_b_x.csv")
h_bias = readdlm("examples/bin/train_b_h.csv")
z_bias = readdlm("examples/bin/train_b_z.csv")
w_vh = readdlm("examples/bin/train_W_xh.csv")
w_hz = readdlm("examples/bin/train_W_zh.csv")

w_vh

