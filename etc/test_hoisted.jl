using Random
using SparseArrays
using Test
using LoopVectorization

function col_contraction_native(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        val = getindex(v, j)
        tot += wij * val
    end
    return tot
end

function col_contraction_reference(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    @inbounds for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * v[j]
    end
    return tot
end

function make_problem(; n = 1000, m = n, density = 0.02, seed = 42, T = Float32)
    rng = MersenneTwister(seed)
    sp = sprand(rng, T, n, m, density)
    v = rand(rng, T, n)
    return v, sp
end

function run_test(; n = 1000, m = n, density = 0.02, seed = 42, T = Float32)
    v, sp = make_problem(; n, m, density, seed, T)

    # Compare all columns; @turbo may reassociate sums, so use tolerance.
    for i in 1:size(sp, 2)
        got = col_contraction_native(v, sp, i)
        ref = col_contraction_reference(v, sp, i)
        @test isapprox(got, ref; rtol = sqrt(eps(T)) * 8, atol = eps(T) * 64)
    end

    println("col_contraction_native test passed for $(size(sp, 1))x$(size(sp, 2)) sparse matrix")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test()
end
