using InteractiveIsing, Processes, LoopVectorization, BenchmarkTools, SparseArrays

g = IsingGraph(30,30)
createProcess(g)
w = @WG "dr -> 1/dr" NN=4
genAdj!(g[1], w)

# params(g).self.val = rand(Float32, 900)
# params(g).b.val = rand(Float32, 900)

# as = getargs(process(g))
# ps = params(g)

# function sum_p_coll(pc::ParamCollection)
#     tot = zero(eltype(pc))
#     @turbo for i in eachindex(pc)
#         tot += pc[i]
#     end
#     return tot
# end

# pc = ParamCollection(ps, :self, :b)

# sum_p_coll(pc)

spvec = sparsevec([Int32(1)],[state(g)[1]], 900)
adj = g.adj

@benchmark $adj*$spvec

function coll_test_turbo(state, adj, i)
    cumsum = zero(eltype(adj))
    @turbo for ptr in nzrange(adj, i)
        j = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        cumsum += wij * state[j]
    end
    return cumsum
end

function coll_test_sparse(vec, adj, i)

    # spvec = sparsevec(([Int32(i)]), [vec[i]], length(vec))
    return sum(vec[i]*adj[:,i])
end

struct SparseSingle{Tv,Ti} <: AbstractSparseVector{Tv,Ti}
    n::Ti
    nzind::Ti
    nzval::Tv
end

sparsesingle(vec::Vector, idx, Ti = Int) = SparseSingle{eltype(vec), Ti}(Ti(length(vec)), Ti(idx), vec[idx])

Base.length(sp::SparseSingle) = sp.n
Base.size(sp::SparseSingle) = (sp.n,)
Base.iterate(sp::SparseSingle, state = 1) = state == 1 ? (sp.nzval, 2) : nothing
Base.getindex(sp::SparseSingle, i) = i == sp.nzind ? sp.nzval : 0
SparseArrays.nonzeros(sp::SparseSingle) = sp.nzval
SparseArrays.nonzeroinds(sp::SparseSingle) = sp.nzind

ssingle = sparsesingle(state(g), 1, Int32)

adj*ssingle


@benchmark coll_test_turbo($state(g), $adj, 1)
@benchmark coll_test_sparse($state(g), $adj, 1)
@benchmark $adj*$spvec

# wref = ParameterRef{:w, (:j,), (:i,)}()
# stateref = ParameterRef{:s, (:j,), ()}()

# con = wref*stateref
# con(as,ps, as.gadj, 1)
