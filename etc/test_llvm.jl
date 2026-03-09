using SparseArrays, LoopVectorization, JLD2

include("src/Utils/StaticFill.jl")

function col_contraction_deferred(v, sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * deferred_getindex(v, j)
    end
    return tot
end

function col_contraction_literal(sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij * 1.0
    end
    return tot
end

m = SparseMatrixCSC(JLD2.load("sparse_50k.jld2", "A"))
sf = StaticFill(1.0, size(m, 2))

# warmup
col_contraction_deferred(sf, m, 1)
col_contraction_literal(m, 1)

using InteractiveUtils

println("=== LLVM IR: deferred_getindex(StaticFill) ===")
display(@code_llvm debuginfo=:none col_contraction_deferred(sf, m, 1))

println("\n\n=== LLVM IR: literal 1.0 ===")
display(@code_llvm debuginfo=:none col_contraction_literal(m, 1))
