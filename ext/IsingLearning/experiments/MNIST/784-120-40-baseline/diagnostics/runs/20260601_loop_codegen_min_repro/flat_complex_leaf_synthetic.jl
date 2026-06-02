# Question: is a FLAT NamedTuple cheap to reconstruct-in-loop even when one leaf
# field has a deeply-parametrized, non-isbits TYPE (like dynamics.model::IsingGraph)?
#
# Layout is flat (scalar s + one leaf g). Only the *type complexity* of g grows.
# If compile time stays flat in depth D  => Fabian is right, flat => no blowup,
#    the stall is elsewhere (constprop / init / IR size).
# If compile time explodes in D         => a complex leaf type in a reconstructed
#    flat NT is itself the codegen hazard.

const RESULTS = joinpath(@__DIR__, "flat_complex_leaf_results.txt")
open(RESULTS, "w") do io; println(io, "depth\tloopA_reconstruct_s\tloopB_invariant_s"); end

# deeply-parametrized, non-isbits leaf type (mimics IsingGraph: nested params + Vectors)
# single recursion => type depth == D, construction O(D)
struct Leaf{A}; a::A; b::Vector{Float64}; payload::Vector{Float64}; tag::Int; end
mkleaf(::Val{0}) = [1.0, 2.0]
mkleaf(::Val{D}) where {D} = Leaf(mkleaf(Val(D - 1)), [1.0], Float64[D], D)

@inline readg(v::Vector{Float64}) = @inbounds v[1]
@inline readg(l::Leaf) = readg(l.a) + l.tag

# A: reconstruct the flat NT each iter (copies g, the deep-typed field), forced inline
@inline stepA(c::NamedTuple) = merge(c, (; s = c.s + readg(c.g) + 1.0))
function loopA(c::NamedTuple, n::Int)
    @inbounds for _ in 1:n
        c = @inline stepA(c)
    end
    return c.s
end

# B: g invariant (separate arg), only scalar carried
@inline stepB(g, s::Float64) = s + readg(g) + 1.0
function loopB(g, s::Float64, n::Int)
    @inbounds for _ in 1:n
        s = @inline stepB(g, s)
    end
    return s
end

function run_for(D::Int)
    g = mkleaf(Val(D))
    cA = (; s = 0.0, g = g)
    tA = @elapsed loopA(cA, 3)
    tB = @elapsed loopB(g, 0.0, 3)
    open(RESULTS, "a") do io
        println(io, "$D\t$(round(tA, digits=3))\t$(round(tB, digits=3))")
    end
    println("depth=$D  reconstruct=$(round(tA,digits=3))s  invariant=$(round(tB,digits=3))s")
    flush(stdout)
end

run_for(1)  # warm infra
for D in (4, 8, 12, 16, 20, 24)
    run_for(D)
end
println("DONE")
