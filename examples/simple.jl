using InteractiveIsing

g = simulate(28, 28, continuous = true)

addLayer!(g, 32,32)

using NPZ, SparseArrays
function weights2sparse(g, ws)
    n_units = size(ws, 1)+size(ws, 2)
    rowval = Int32[]
    colval = Int32[]
    nzval = Float32[]
    sizehint!(rowval, n_units)
    sizehint!(colval, n_units)
    sizehint!(nzval, n_units)

    for i in 1:size(ws, 2)
        for j in 1:size(ws, 1)
                col = InteractiveIsing.idxLToG(j, g[1])
                row = InteractiveIsing.idxLToG(i, g[2])

                push!(rowval, row)
                push!(colval, col)
                push!(rowval, col)
                push!(colval, row)
                push!(nzval, ws[j,i])
                push!(nzval, ws[j,i])
        end
    end
    return sparse(rowval, colval, nzval, n_units, n_units)
end

sp_adj(g, weights2sparse(g, npzread("/Users/fabian/Downloads/bm_data/rbm_W_finetuned.npy")))

resetstate(g) = state(g) .= rand(Float32, length(state(g)))

weights12_2_sparse(w1, w2)