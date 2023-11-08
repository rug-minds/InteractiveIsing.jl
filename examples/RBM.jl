using InteractiveIsing

g = simulate(28, 28, type = Continuous, set = (0f0, 1f0))
addLayer!(g,32,32, type = Discrete, set = (0f0, 1f0))
addLayer!(g, 2, 5, type = Discrete, set = (0f0, 1f0))

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
