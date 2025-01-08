using InteractiveIsing
include("SDefects.jl")

# all_coords = [(i,j,1) for i in 1:40, j in 1:40]
# all_idxs = [1:2*40^2;]
# add_sdefects!(g[1], 0.1, all_idxs...)
# layer1defects = get_sdefects(g[1]).data[1:1600]
# layer2defects = get_sdefects(g[1]).data[1601:3200]
# jump!.(layer2defects, 0,0,1)
# jump!.(layer1defects, 0,0,1)


all_coords = [(1,j,k) for j in 1:40, k in 1:40]
add_sdefects!(g[1], 0.1, all_coords...)
yz1_defects = get_sdefects(g[1]).data[1:1600]
jump!.(yz1_defects, -1,0,0)