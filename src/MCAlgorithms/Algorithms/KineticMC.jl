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

@inline function _proposal_for_index(proposer, rng, i::Integer)
    i = Int(i)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[i]
    layer_idx = spin_idx_to_layer_idx(i, proposer.layers)
    newstate = @inline inline_layer_dispatch(layer -> randstate(rng, layer, oldstate), layer_idx, proposer.layers)
    return FlipProposal{statetype(proposer)}(i, oldstate, newstate, layer_idx, false)
end

"""Return a non-accepted diagnostic proposal for steps with no drawable event."""
@inline function _inactive_flip_proposal(proposer::P) where {P}
    SType = statetype(proposer)
    return FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
end

@inline function _rate_from_delta(r0::T, ΔE::T, t::T) where {T}
    if !(t > zero(T))
        # At T=0, only accept transitions that lower energy
        return ΔE <= zero(T) ? r0 : zero(T)
    end

    ΔE <= zero(T) && return r0
    exponent = clamp(-ΔE / t, T(-80), zero(T))
    r = r0 * exp(exponent)
    if !isfinite(r) || r < zero(T)
        return zero(T)
    end
    return r
end

@inline function FlipEnergies(model, n::Int, r0::T) where {T}
    ΔEs = zeros(T, n)
    rates = zeros(T, n)
    # Targets are filled by rebuild_rates!/refresh_rate!, so avoid copying the graph.
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

@inline function refresh_rate!(fe::FlipEnergies{T}, context, i::Integer, t::T) where {T}
    i = Int(i)
    (;model, hamiltonian, proposer, rng) = context

    @inline proposal = _proposal_for_index(proposer, rng, i)
    @inline ΔE = calculate(ΔH(), hamiltonian, model, proposal)
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
    (;model, hamiltonian, proposer, rng) = context
    totalrate = zero(eltype(fe))

    for i in eachindex(fe.rates)
        proposal = _proposal_for_index(proposer, rng, i)
        ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal)
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

@inline function update_local_rates!(fe::FlipEnergies, context, j::Integer, t)
    j = Int(j)
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
    package(CompositeAlgorithm(kinetic, destr, Route(destr => kinetic, :isinggraph => :model)))
end

export KineticMC

@inline update!(::KineticMC, hterm, model, proposal) = update!(Metropolis(), hterm, model, proposal)

@inline function _kinetic_active_count(model)
    active = @inline sampling_indices(model)
    return length(active)
end

"""
    _kinetic_log_proposal_ratio(proposer, proposal)

Return `log(q(reverse) / q(forward))` for the proposal kernel.

The built-in Ising proposers used here are symmetric, so the default is zero.
Non-symmetric custom proposers must specialize this method for `KineticMC` to
be Boltzmann-correct.
"""
@inline _kinetic_log_proposal_ratio(proposer, proposal) = zero(eltype(proposal))

@inline function _kinetic_accept(ΔE::T, temperature::T, log_q_ratio::T, rng) where {T}
    if !(temperature > zero(T))
        return ΔE <= zero(T)
    end

    log_acceptance = -ΔE / temperature + log_q_ratio
    return log_acceptance >= zero(T) || log(rand(rng, T)) < log_acceptance
end

@inline function StatefulAlgorithms.init(algo::KineticMC, context::Cont) where {Cont}
    (;model) = context

    hamiltonian = model.hamiltonian
    rng = Random.MersenneTwister()
    hamiltonian = init!(hamiltonian, model)
    proposer = get_proposer(model)

    T = eltype(model)
    time = Ref(zero(T))
    active_count = Ref(@inline _kinetic_active_count(model))

    returnargs = (;model, hamiltonian, proposer, rng, time, active_count)
    return returnargs
end

@inline function StatefulAlgorithms.step!(kinetic::KineticMC, context::C) where {C}
    (;model, rng, proposer) = context

    T = eltype(model)
    t = T(temp(model))

    if @inline consume_changed!(model)
        context.active_count[] = @inline _kinetic_active_count(model)
    end

    totalrate = T(kinetic.r0) * T(context.active_count[])
    if !(totalrate > zero(T)) || !isfinite(totalrate)
        dt = zero(T)
        yield()
        proposal = @inline _inactive_flip_proposal(proposer)
        return (; j = 0, ΔE = zero(T), dt, totalrate = zero(T),
            proposal, kmc_time = context.time[],
            accepted = 0, attempted = 1)
    end

    proposal = @inline rand(rng, proposer)
    ΔE = T(@inline calculate(ΔH(), context.hamiltonian, model, proposal))
    log_q_ratio = T(@inline _kinetic_log_proposal_ratio(proposer, proposal))
    accept_event = @inline _kinetic_accept(ΔE, t, log_q_ratio, rng)
    if accept_event
        proposal = @inline accept(proposer, proposal)
        @inline update!(kinetic, context.hamiltonian, model, proposal)
    end

    dt = -log(max(rand(rng, T), eps(T))) / totalrate
    context.time[] += dt

    return (;j = at_idx(proposal), ΔE, dt, totalrate, proposal,
        kmc_time = context.time[], accepted = Int(accept_event), attempted = 1)
end
