using InteractiveIsing

g = simulate(28, 28, type = Continuous, set = (0f0, 1f0))
addLayer!(g,32,32, type = Discrete, set = (0f0, 1f0))

function weights_2_sparse(w1)
    nstates = length(state(g))
    n_conns = length(w1)
    rowval = Int32[]
    colval = Int32[]
    nzval = Float32[]
    sizehint!(rowval, n_conns)
    sizehint!(colval, n_conns)
    sizehint!(nzval, n_conns)

    for i in 1:size(w1, 2)
        for j in 1:size(w1, 1)
                col = InteractiveIsing.idxLToG(j, g[1])
                row = InteractiveIsing.idxLToG(i, g[2])

                push!(rowval, row)
                push!(colval, col)
                push!(rowval, col)
                push!(colval, row)
                push!(nzval, w1[j,i])
                push!(nzval, w1[j,i])
        end
    end

    return sparse(rowval, colval, nzval, nstates, nstates)
end

using NPZ, SparseArrays
w = npzread("/Users/fabian/Downloads/bm_data/rbm_W_finetuned.npy")
sp_adj(g, weights_2_sparse(w))
createAvgWindow(g[1])
# using GLMakie, DataStructures
# # scene = Scene(resolution = (800, 800));
# f = Figure();
# ax = Axis(f[1, 1], aspect = 1)
# d = display(GLMakie.Screen(), f)


# const shitload = Matrix{AverageCircular{Float32}}(undef, 28, 28)
# for idx in eachindex(shitload)
#     shitload[idx] = AverageCircular(Float32, 1024)
# end
# avgs = zeros(Float32, 28, 28)
# const img_ob = Observable(avgs)
# image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)



# function updateshitload(shitload, state)
#     for idx in eachindex(shitload)
#         push!(shitload[idx], state[idx])
#     end
# end

# function updateavgs(avgs, shitload)
#     for idx in eachindex(avgs)
#         avgs[idx] = avg(shitload[idx])
#     end
# end

# const tShouldRun = Ref(true)

# function update(g, shitload, avgs, img_ob)
#     t = Threads.@spawn begin
#         updateshitload(shitload, state(g[1]))
#         updateavgs(avgs, shitload)
#     end
#     wait(t)
#     notify(img_ob)
# end

# timer = Timer(t -> update(g, shitload, avgs, img_ob), 0, interval = 1/1024)