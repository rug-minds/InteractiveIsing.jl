struct FenwickTree{T}
    tree::Vector{T}
end

function FenwickTree(n::Int, T=Float64)
    FenwickTree(zeros(T, n))
end

function update!(ft::FenwickTree, i, delta)
    while i <= length(ft.tree)
        ft.tree[i] += delta
        i += i & -i
    end
end

function prefix_sum(ft::FenwickTree, i::Int)
    s = 0.0
    while i > 0
        s += ft.tree[i]
        i -= i & -i
    end
    return s
end

function total_sum(ft::FenwickTree)
    return prefix_sum(ft, length(ft.tree))
end

# Find index such that prefix_sum(index) >= u
function find_index(ft::FenwickTree, u)
    idx = 0
    mask = 1 << (floor(Int, log2(length(ft.tree))))
    s = 0.0
    while mask != 0
        t = idx + mask
        if t <= length(ft.tree) && s + ft.tree[t] < u
            s += ft.tree[t]
            idx = t
        end
        mask >>= 1
    end
    # Clamp to last valid index if u >= total sum (due to floating point error)
    return min(idx + 1, length(ft.tree))
end

mutable struct FlipEnergies{T}
    ΔEs::Vector{T}
    rates::Vector{T}
    fenwick::FenwickTree{T} # Fenwick tree for cumulative rates
    totalrate::Float64
    r0::T
end

Base.eltype(::Type{FlipEnergies{T}}) where T = T

function get_r(fe::FlipEnergies, β, i)
    r = fe.r0 * exp(-β * fe.ΔEs[i])
    return r
end

function FlipEnergies(g::AbstractIsingGraph, args::As, r0 = one(eltype(g))) where As
    totalrate = zero(eltype(g.state))
    vec = zeros(eltype(g.state), length(g.state))
    rates = zeros(eltype(g.state), length(g.state))
    fe = FlipEnergies(vec, rates, FenwickTree(length(g.state), eltype(g.state)), Float64(totalrate), r0)
    for i in eachindex(fe.ΔEs)
       init_i!(fe, g, args, i)
    end
    return fe
end

function init_i!(fe::FlipEnergies, g, args, i)
    (;gstate, gadj, deltafunc, lmeta, rng) = args
    newstate = SparseVal((@inline sampleState(statetype(lmeta), gstate[i], rng, stateset(lmeta))), Int32(length(gstate)), Int32(i))
    fe.ΔEs[i] = deltafunc((;g, args..., newstate), j = i)
    previousrate = fe.rates[i]
    t = temp(g)
    if t == 0
        t = eps(eltype(g.state)) # Avoid division by zero
    end
    maxexp = 700 # Prevent overflow in exp
    exponent = clamp(-fe.ΔEs[i] / t, -maxexp, maxexp)
    r = fe.r0 * exp(exponent)
    if !isfinite(r) || r < 0
        r = zero(eltype(g.state))
    end
    delta = r - previousrate
    if r > 0
        fe.rates[i] = r
        # fe.totalrate += delta
        update!(fe.fenwick, i, delta)
    else
        fe.rates[i] = zero(eltype(g.state))
        # fe.totalrate -= previousrate
        update!(fe.fenwick, i, -previousrate)
    end
    return fe
end

function recalc_temp!(fe::FlipEnergies, args)
    for i in eachindex(fe.ΔEs)
        init_i!(fe, args.g, args, i)
    end
    return fe
end

function recalc!(fe::FlipEnergies, args, j)
    (;gadj) = args
    connections = gadj.rowval[nzrange(gadj, j)]
    for i in connections
        init_i!(fe, args.g, args, i)
    end
    return fe
end

function get_totalrate(fe::FlipEnergies)
    cum = zero(eltype(fe))
   @turbo for i in eachindex(fe.rates)
        cum += fe.rates[i]
    end
    fe.totalrate = cum
    return fe.totalrate
end

function get_u(rng, fe)
    totalrate = get_totalrate(fe)
    if totalrate <= 0 || isnan(totalrate)
        fe.totalrate = zero(eltype(fe))
        return zero(eltype(fe))
    else
        # println("totalrate: ", totalrate)
        u = Uniform(zero(eltype(fe)), eltype(fe)(totalrate))
        return rand(rng, u)
    end
    # if fe.totalrate <= 0 || isnan(fe.totalrate)
    #     fe.totalrate = zero(eltype(fe))
    #     return zero(eltype(fe))
    # else
    #     u = Uniform(zero(eltype(fe)), eltype(fe)(fe.totalrate))
    #     return rand(rng, u)
    # end
end

struct KineticMC <: MCAlgorithm end
export KineticMC

function Processes.init(::KineticMC, args::As) where As
    (;g) = args
    gstate = g.state
    gadj = g.adj
    params = g.params
    iterator = ising_it(g)
    hamiltonian = init!(g.hamiltonian, g)
    deltafunc = deltaH(hamiltonian)
    rng = Random.GLOBAL_RNG
    M = Ref(sum(g.state))
    Δs_j = Ref(zero(eltype(g.state)))

    lmeta = LayerMetaData(g[1])
    lasttemp = Ref(temp(g))

    

    args = (;gstate, gadj, params, iterator, hamiltonian, deltafunc, lmeta, rng, M, Δs_j, lasttemp, lastdt)
    ΔEs = FlipEnergies(g, args)
    return (;args..., ΔEs)
end


@inline function (::KineticMC)(@specialize(args))
    (;g, gstate, ΔEs, rng) = args

    if temp(g) != args.lasttemp[]
        recalc_temp!(ΔEs, args)
        args.lasttemp[] = temp(g)
    end
    β = one(eltype(g))/(temp(g))
    u = get_u(rng, ΔEs)
    totalrate = get_totalrate(ΔEs)
    u = min(u, totalrate - eps(totalrate))
    if u <= 0
        return nothing # No valid flip found
    end
    j = find_index(ΔEs.fenwick, u)

    g.state[j] = -g.state[j] # Flip the state
    # @hasarg if M isa Ref
    #     M[] += (g.state[j] - gstate[j])
    # end|
    # @hasarg if Δs_j isa Ref
    #     Δs_j[] = g.state[j] - gstate[j]
    # end

    recalc!(ΔEs, args, j)
    # @inline update!(args.hamiltonian, args)
    dt = 
    return
end

