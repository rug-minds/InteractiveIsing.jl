using SparseArrays, LoopVectorization, BenchmarkTools, JLD2

include("src/Utils/StaticFill.jl")

# --- native v[j] path (memory op) ---
function col_contraction_native(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * v[j]
    end
    return tot
end

# --- deferred_getindex path (compute op) ---
function col_contraction_deferred(v, sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * deferred_getindex(v, j)
    end
    return tot
end

# --- literal baseline ---
function col_contraction_literal(sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij * 1.0
    end
    return tot
end

# --- load data ---
m = SparseMatrixCSC(JLD2.load("sparse_50k.jld2", "A"))
v = rand(size(m, 2))
sf = StaticFill(1.0, size(m, 2))
const col = 1

# --- warmup ---
col_contraction_native(v, m, col)
col_contraction_deferred(v, m, col)
col_contraction_deferred(sf, m, col)
col_contraction_literal(m, col)

# --- correctness ---
println("=== Correctness ===")
r1 = col_contraction_native(v, m, col)
r2 = col_contraction_deferred(v, m, col)
r3 = col_contraction_deferred(sf, m, col)
r4 = col_contraction_literal(m, col)
println("native(Vec)       = $r1")
println("deferred(Vec)     = $r2,  match native = $(isapprox(r1, r2))")
println("deferred(SF)      = $r3")
println("literal           = $r4,  match SF     = $(isapprox(r3, r4))")

# --- benchmarks ---
println("\n=== Benchmarks (sparse col_contraction, col=$col) ===")
print("Native v[j] (Vector):       "); display(@benchmark col_contraction_native($v, $m, $col))
print("Deferred (Vector):          "); display(@benchmark col_contraction_deferred($v, $m, $col))
print("Deferred (StaticFill):      "); display(@benchmark col_contraction_deferred($sf, $m, $col))
print("Literal 1.0:                "); display(@benchmark col_contraction_literal($m, $col))
