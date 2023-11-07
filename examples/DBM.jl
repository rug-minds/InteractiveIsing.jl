using InteractiveIsing

g = simulate(28, 28, type = Continuous, set = (0f0, 1f0))
addLayer!(g,16,32, type = Discrete, set = (0f0, 1f0))
addLayer!(g,32,32, type = Discrete, set = (0f0, 1f0))
# addLayer!(g, 40,40, type = Continuous)
# wg = @WG "(dr) -> 1" NN=1
# coords

function weights12_2_sparse(w1, w2)
    nstates = length(state(g))
    n_conns = length(w1)+length(w2)
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
    for i in 1:size(w2, 2)
        for j in 1:size(w2, 1)
                col = InteractiveIsing.idxLToG(j, g[2])
                row = InteractiveIsing.idxLToG(i, g[3])

                push!(rowval, row)
                push!(colval, col)

                push!(rowval, col)
                push!(colval, row)

                push!(nzval, w2[j,i])
                push!(nzval, w2[j,i]) 
        end
    end
    return sparse(rowval, colval, nzval, nstates, nstates)
end

using NPZ, SparseArrays
w12= npzread("/Users/fabian/Downloads/bm_data/dbm_W1_finetuned.npy")
w23 = npzread("/Users/fabian/Downloads/bm_data/dbm_W2_finetuned.npy")
sp_adj(g, weights12_2_sparse(npzread("/Users/fabian/Downloads/bm_data/dbm_W1_finetuned.npy"), npzread("/Users/fabian/Downloads/bm_data/dbm_W2_finetuned.npy")))

using GLMakie, DataStructures
# scene = Scene(resolution = (800, 800));
f = Figure();
ax = Axis(f[1, 1], aspect = 1)
d = display(GLMakie.Screen(), f)


const shitload = Matrix{CircularBuffer{Float32}}(undef, 28, 28)
for idx in eachindex(shitload)
    shitload[idx] = CircularBuffer{Float32}(512)
end
avgs = zeros(Float32, 28, 28)
const im_obs = Observable(avgs)
image!(ax, im_obs, colormap = :thermal, fxaa = false, interpolate = false)



function updateshitload(shitload, state)
    for idx in eachindex(shitload)
        push!(shitload[idx], state[idx])
    end
end

function updateavgs(avgs, shitload)
    for idx in eachindex(avgs)
        avgs[idx] = sum(shitload[idx])/length(shitload[idx])
    end
end

const tShouldRun = Ref(true)

tsr() = tShouldRun[] = !tShouldRun[]

function update(g, shitload, avgs, im_obs)
    updateshitload(shitload, state(g[1]))
    updateavgs(avgs, shitload)
    im_obs[] = avgs
end

timer = Timer(t -> update(g, shitload, avgs, im_obs), 0, interval = 1/256)
