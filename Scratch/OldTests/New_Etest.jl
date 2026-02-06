using LoopVectorization, BenchmarkTools, Metal
const st1 = state(g)
const adj1 = adj(g)
const tuples1 = adj1[1]

tuples2matrix(tuples) = [tuples[i][j] for i in eachindex(tuples), j in 1:2]



const matrix1 = tuples2matrix(tuples1) 

function efac(state::Vector{Float32}, conns::Vector{Tuple{Int32,Float32}})
    efac = 0f0
    @inbounds @simd for idx in eachindex(conns)
        efac += -state[first(conns[idx])] * last(conns[idx])
    end
    return efac
end

function efacsimd(state::Vector{Float32}, conns::Vector{Tuple{Int32,Float32}})
    efac = 0f0
    lane = VecRange{2}(0)
    @inbounds for idx in eachindex(conns)
        efac += state[conns[lane+idx][1]] * conns[lane+idx][2]
    end
    return efac
end

function efac(state::Vector{Float32}, conns::Matrix)
    efac = 0f0
    @inbounds @simd for idx in 1:size(conns,1)
        efac += state[conns[idx,1]] * conns[idx,2]
    end

    return efac
end

struct Connections
    idxs::Vector{Int32}
    weights::Vector{Float32}
end

vectuple2conns(v::Vector{Tuple{Int32,Float32}}) = Connections([x[1] for x in v], [x[2] for x in v])

const conn1 = vectuple2conns(tuples1)
const conns2 = [vectuple2conns(adj1[i]) for i in eachindex(adj1)]

function efac(state::Vector{Float32}, conns::Connections)
    efac = 0f0
    idxs = conns.idxs
    weights = conns.weights
    @turbo for idx in eachindex(idxs)
        efac += state[idxs[idx]] * weights[idx]
    end
    return efac
end

const mtl_state = Metal.zeros(Float32, length(state(g)); storage = Shared)
const cpu_state = unsafe_wrap(Vector{Float32}, mtl_state, length(state(g)))
const mtl_adj1 = Metal.zeros(Tuple{Int32,Float32}, length(adj(g)[1]), storage = Shared)
const cpu_adj1 = unsafe_wrap(Vector{Tuple{Int32,Float32}}, mtl_adj, length(adj(g)[1]))
const mtl_mult_result = Metal.zeros(Float32, length(adj(g)[1]), storage = Shared)
const cpu_mult_result = unsafe_wrap(Vector{Float32}, mtl_mult_result, length(adj(g)[1]))

function efac_kernel(mtl_state, mtl_adj, mtl_mult_result)
    i = thread_position_in_grid_1d()
    if i <= length(mtl_adj)
        tupl = mtl_adj[i]
        idx = tupl[1]
        weight = tupl[2]
        mtl_mult_result[i] = -mtl_state[idx] * weight
    end
    return nothing
end

function efacMetal(mtl_state, mtl_adj, mtl_mult_result, cpu_mult_result)
    @metal threads=1024 groups = 10 efac_kernel(mtl_state, mtl_adj, mtl_mult_result)
    return Float32(reduce(+, mtl_mult_result))
end

using SparseArrays

function tuples2sparse(adj)
    colidx_len = 0
    for col in adj
        colidx_len += length(col)
    end
    colidx = Vector{Int32}(undef, colidx_len)
    colidxidx = 1
    for (idx,col) in enumerate(adj)
        for i in 1:length(col)
            colidx[colidxidx] = Int32(idx)
            colidxidx += 1
        end
    end

    rowidx = Vector{Int32}(undef, colidx_len)
    rowidxidx = 1
    for (idx,col) in enumerate(adj)
        for i in 1:length(col)
            rowidx[rowidxidx] = Int32(adj[idx][i][1])
            rowidxidx += 1
        end
    end

    vals = Vector{Float32}(undef, colidx_len)
    valsidx = 1
    for (idx,col) in enumerate(adj)
        for i in 1:length(col)
            vals[valsidx] = adj[idx][i][2]
            valsidx += 1
        end
    end
    return sparse(rowidx, colidx, vals)
end
const sp1 = tuples2sparse(adj1)

nzrange(sp1,1)

sp1[1,:]

function efac(state, sparse::SparseMatrixCSC, idx, stype)
    efac = 0f0 
    @turbo for idx in nzrange(sparse, idx)
        efac += state[sparse.rowval[idx]] * sparse.nzval[idx]
    end
    return efac
end

function loop_sparse(state, sparse)
    for _ in 1:1000

        idx = rand(1:(size(sparse,1)))
        efac(state, sparse, idx)
    end
end

function loop_adj(state, adj)
    for _ in 1:1000
        idx = rand(1:length(adj))
        @inbounds efac(state, adj[idx])
    end
end

function loop_conns(state, conns)
    for _ in 1:1000
        idx = rand(1:length(conns))
        efac(state, conns[idx])
    end
end