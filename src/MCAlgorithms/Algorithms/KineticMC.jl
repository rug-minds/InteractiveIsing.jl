struct FenwickTree{T}
    tree::Vector{T}
end

@inline FenwickTree(n::Int, T = Float64) = FenwickTree(zeros(T, n))

@inline function update!(ft::FenwickTree{T}, i::Int, delta::T) where {T}
    n = length(ft.tree)
    while i <= n
        @inbounds ft.tree[i] += delta
        i += i & -i
    end
    return ft
end

@inline function prefix_sum(ft::FenwickTree{T}, i::Int) where {T}
    s = zero(T)
    while i > 0
        @inbounds s += ft.tree[i]
        i -= i & -i
    end
    return s
end

@inline total_sum(ft::FenwickTree) = prefix_sum(ft, length(ft.tree))

@inline function find_index(ft::FenwickTree{T}, u::T) where {T}
    n = length(ft.tree)
    n == 0 && return 0

    idx = 0
    s = zero(T)
    mask = prevpow(2, n)

    while mask != 0
        t = idx + mask
        if t <= n
            v = @inbounds ft.tree[t]
            if s + v < u
                s += v
                idx = t
            end
        end
        mask >>= 1
    end

    return min(idx + 1, n)
end

mutable struct FlipEnergies{T}
    ΔEs::Vector{T}
    rates::Vector{T}
    targets::Vector{T}
    fenwick::FenwickTree{T}
    totalrate::T
    r0::T
end

@inline Base.eltype(::Type{FlipEnergies{T}}) where {T} = T

@inline function _proposal_for_index(proposer, rng, i::Int)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[i]
    layer_idx = spin_idx_to_layer_idx(i, proposer.layers)
    newstate = @inline inline_layer_dispatch(layer -> randstate(rng, layer, oldstate), layer_idx, proposer.layers)
    return FlipProposal{statetype(proposer)}(i, oldstate, newstate, layer_idx, false)
end

@inline function _rate_from_delta(r0::T, ΔE::T, t::T) where {T}
    if !(t > zero(T))
        # At T=0, only accept transitions that lower energy
        return ΔE < zero(T) ? r0 : zero(T)
    end

    exponent = clamp(-ΔE / t, T(-700), T(700))
    r = r0 * exp(exponent)
    if !isfinite(r) || r < zero(T)
        return zero(T)
    end
    return r
end

@inline function FlipEnergies(state, n::Int, r0::T) where {T}
    ΔEs = zeros(T, n)
    rates = zeros(T, n)
    # Targets are filled by rebuild_rates!/refresh_rate!, so avoid copying state.
    targets = Vector{T}(undef, n)
    return FlipEnergies(ΔEs, rates, targets, FenwickTree(n, T), zero(T), r0)
end

@inline function build_fenwick_from_rates!(ft::FenwickTree{T}, rates::Vector{T}) where {T}
    tree = ft.tree
    copyto!(tree, rates)
    n = length(tree)
    @inbounds for i in 1:n
        j = i + (i & -i)
        if j <= n
            tree[j] += tree[i]
        end
    end
    return ft
end

@inline function refresh_rate!(fe::FlipEnergies{T}, context, i::Int, t::T) where {T}
    (;state, hamiltonian, proposer, rng) = context

    @inline proposal = _proposal_for_index(proposer, rng, i)
    @inline ΔE = calculate(ΔH(), hamiltonian, state, proposal)
    @inline r = _rate_from_delta(fe.r0, ΔE, t)

    @inbounds oldr = fe.rates[i]
    delta = r - oldr

    @inbounds fe.ΔEs[i] = ΔE
    @inbounds fe.rates[i] = r
    @inbounds fe.targets[i] = to_val(proposal)
    @inline update!(fe.fenwick, i, delta)

    fe.totalrate += delta
    nothing
end

@inline function rebuild_rates!(fe::FlipEnergies, context, t)
    (;state, hamiltonian, proposer, rng) = context
    totalrate = zero(eltype(fe))

    for i in eachindex(fe.rates)
        proposal = _proposal_for_index(proposer, rng, i)
        ΔE = @inline calculate(ΔH(), hamiltonian, state, proposal)
        r = _rate_from_delta(fe.r0, ΔE, t)

        @inbounds fe.ΔEs[i] = ΔE
        @inbounds fe.rates[i] = r
        @inbounds fe.targets[i] = to_val(proposal)
        totalrate += r
    end

    fe.totalrate = totalrate
    build_fenwick_from_rates!(fe.fenwick, fe.rates)
    return fe
end

@inline function update_local_rates!(fe::FlipEnergies, context, j::Int, t)
    (;adj) = context

    refresh_rate!(fe, context, j, t)
    rowvals = SparseArrays.rowvals(adj)
    for ptr in SparseArrays.nzrange(adj, j)
        i = @inbounds rowvals[ptr]
        i == j && continue
        refresh_rate!(fe, context, i, t)
    end
    return fe
end

@inline function draw_event_index(rng, fe::FlipEnergies{T}) where {T}
    totalrate = fe.totalrate
    if !(totalrate > zero(T)) || !isfinite(totalrate)
        return 0, zero(T)
    end

    # Keep u strictly positive to avoid selecting index 0 from finite-precision edge cases.
    u = max(rand(rng, T) * totalrate, eps(T))
    u = min(u, totalrate)
    j = find_index(fe.fenwick, u)
    return j, totalrate
end

struct KineticMC <: IsingMCAlgorithm
    r0::Float64
end

@inline KineticMC(; r0 = 1.0) = KineticMC(Float64(r0))

@inline function IsingKinetic(; r0 = 1.0)
    destr = DestructureInput()
    kinetic = KineticMC(r0 = r0)
    package(SimpleAlgo(kinetic, destr, Route(destr => kinetic, :isinggraph => :structure)))
end

export KineticMC

@inline update!(::KineticMC, hterm, state, proposal) = update!(Metropolis(), hterm, state, proposal)

@inline function Processes.init(algo::KineticMC, context::Cont) where {Cont}
    (;structure) = context

    state = structure
    adj = InteractiveIsing.adj(structure)
    hamiltonian = structure.hamiltonian
    rng = Random.MersenneTwister()
    hamiltonian = init!(hamiltonian, structure)
    proposer = get_proposer(structure)
    proposal = @inline rand(rng, proposer)

    T = eltype(state)
    t = T(temp(structure))
    rates = FlipEnergies(state, InteractiveIsing.nstates(state), T(algo.r0))
    lasttemp = Ref(t)
    lastdt = Ref(zero(T))
    j = 0
    ΔE = zero(T)
    dt = zero(T)
    totalrate = zero(T)
    kinetic_context = (;structure, state, adj, hamiltonian, proposer, rng)
    rebuild_rates!(rates, kinetic_context, t)

    returnargs = (;kinetic_context..., proposal, rates, lasttemp, lastdt, j, ΔE, dt, totalrate)
    return returnargs
end

@inline function Processes.step!(kinetic::KineticMC, context::C) where {C}
    (;structure, state, rates, rng, proposer) = context

    t = eltype(state)(temp(structure))
    lasttemp = context.lasttemp[]
    if isfinite(lasttemp) && abs(t - lasttemp) > eps(eltype(state)) * max(abs(t), abs(lasttemp), one(eltype(state)))
        rebuild_rates!(rates, context, t)
        context.lasttemp[] = t
    elseif !isfinite(lasttemp)
        rebuild_rates!(rates, context, t)
        context.lasttemp[] = t
    end

    j, totalrate = draw_event_index(rng, rates)
    if j == 0
        context.lastdt[] = zero(eltype(state))
        return (;j = 0, ΔE = zero(eltype(state)), dt = context.lastdt[], totalrate = zero(eltype(state)), proposal = context.proposal)
    end

    spins = @inline InteractiveIsing.state(state)
    oldstate = @inbounds spins[j]
    layer_idx = spin_idx_to_layer_idx(j, proposer.layers)
    proposal = FlipProposal{eltype(state)}(j, oldstate, @inbounds(rates.targets[j]), layer_idx, false)
    proposal = @inline accept(proposer, proposal)
    ΔE = @inbounds rates.ΔEs[j]

    update_local_rates!(rates, context, j, t)

    dt = -log(max(rand(rng, eltype(state)), eps(eltype(state)))) / totalrate
    context.lastdt[] = dt

    @inline update!(kinetic, context.hamiltonian, state, proposal)

    return (;j, ΔE, dt, totalrate, proposal)
end
