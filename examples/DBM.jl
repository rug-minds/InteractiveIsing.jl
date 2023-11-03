using InteractiveIsing

g = simulate(28, 28, continuous = true)
addLayer!(g,16,32, type = Discrete)
addLayer!(g,32,32, type = Discrete)
addLayer!(g,40,40, type = Continuous)

function weights12_2_sparse(w1, w2)
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
    return sparse(rowval, colval, nzval, n_conns, n_conns)
end

using NPZ, SparseArrays
sp_adj(g, weights12_2_sparse(npzread("/Users/fabian/Downloads/bm_data/rbm_W_finetuned.npy"), npzread("/Users/fabian/Downloads/bm_data/rbm_W_finetuned.npy")))