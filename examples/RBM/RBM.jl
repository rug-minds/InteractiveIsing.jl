using InteractiveIsing

g = IsingGraph(precision=Float64, architecture=[(28, 28, Discrete), (32, 32, Discrete)], sets=[(0, 1), (0, 1)])
simulate(g)

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
            push!(nzval, w1[j, i])
            push!(nzval, w1[j, i])
        end
    end

    return sparse(rowval, colval, nzval, nstates, nstates)
end

using JLD2, SparseArrays
d = load("examples/RBM/rbm1.jld2")
w = reshape(d["weights"], (28 * 28, 32 * 32))
biases = [d["vbias"][:]; d["hbias"][:]]
adj(g, weights_2_sparse(w))
setParam!(g, :b, biases, true)
# bfield(g) .= biases
#TODO:MOVE THIS TO PARAMS
# setSType!(g, :Magfield => true)
w2 = LayerWindow(g[2])
set_colorrange(g[1])
restart(g)