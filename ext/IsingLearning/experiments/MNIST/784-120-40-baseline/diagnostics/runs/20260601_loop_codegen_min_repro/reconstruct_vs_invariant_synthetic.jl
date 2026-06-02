# Self-contained test: does compile cost track "big aggregate as a value"
# (Fabian's claim: f(s1,s2) == f(s{s1,s2}), bundling is free) or does it track
# "big aggregate RECONSTRUCTED + loop-carried each iteration" (Claude's claim)?
#
# Same data, same reads in both. The ONLY difference:
#   A = the varying scalar is written back into ONE aggregate that also holds the
#       invariant payload, so the whole N+1-field thing is rebuilt every iter and
#       becomes a loop-carried phi  (this is today's ProcessContext merge).
#   B = invariant payload is a separate argument, never reassigned; only the
#       scalar is loop-carried.      (this is the env/carry projection)
#
# If A and B compile in the same time => bundling is irrelevant, Fabian is right.
# If A grows superlinearly while B stays flat => reconstruction is the axis.

const RESULTS = joinpath(@__DIR__, "reconstruct_vs_invariant_results.txt")
open(RESULTS, "w") do io; println(io, "N\tloopA_compile_s\tloopB_compile_s"); end

makepayload(::Val{N}) where {N} =
    NamedTuple{ntuple(i -> Symbol("f", i), N)}(ntuple(i -> iseven(i) ? Float64(i) : Int32(i), N))

@inline _read(nt::NamedTuple) = Float64(nt[1]) + Float64(nt[2])

# A: one aggregate holding scalar s + payload; rebuilt every iteration
function loopA(c::NamedTuple, n::Int)
    @inbounds for _ in 1:n
        c = merge(c, (; s = c.s + _read(c) + 1.0))
    end
    return c.s
end

# B: payload invariant (separate arg, never reassigned); only s loop-carried
function loopB(payload::NamedTuple, s::Float64, n::Int)
    @inbounds for _ in 1:n
        s = s + _read(payload) + 1.0
    end
    return s
end

function run_for(N::Int)
    payload = makepayload(Val(N))
    cA = merge((; s = 0.0), payload)
    # first call pays full specialization + LLVM codegen for this concrete N
    tA = @elapsed loopA(cA, 3)
    tB = @elapsed loopB(payload, 0.0, 3)
    open(RESULTS, "a") do io
        println(io, "$N\t$(round(tA, digits=3))\t$(round(tB, digits=3))")
    end
    println("N=$N  loopA=$(round(tA,digits=3))s  loopB=$(round(tB,digits=3))s")
    flush(stdout)
    return nothing
end

# warm the generic infra (merge, _read) at tiny size so it isn't charged to N=8
run_for(2)
for N in (8, 16, 32, 64, 128, 192, 256)
    run_for(N)
end
println("DONE")
