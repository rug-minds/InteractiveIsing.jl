export KineticMonteCarloLoop, IsingKineticMonteCarloLoop

"""
    KineticMonteCarloLoop(; r0=1.0, full_refresh_interval=0, rate_rule=:metropolis)

Continuous-time single-spin kinetic Monte Carlo loop using the `Processes`
`init`/`step!` interface.

Each `Processes.step!` selects one active spin event with probability
proportional to its cached transition rate, accepts it directly, advances the
internal clock by an exponentially distributed waiting time, and refreshes the
selected spin plus its graph neighbours. The default `:metropolis` rate is
`r0 * min(1, exp(-ΔE / T))`, matching the stationary distribution targeted by
`Metropolis`. `:arrhenius` keeps the downhill `exp(-ΔE / T)` speedup.

This file is intentionally not included from `Algorithms.jl` yet.
"""
struct KineticMonteCarloLoop{RateRule,T<:Real} <: IsingMCAlgorithm
    r0::T
    full_refresh_interval::Int

    function KineticMonteCarloLoop{RateRule,T}(r0::T, full_refresh_interval::Int) where {RateRule,T<:Real}
        RateRule === :metropolis || RateRule === :arrhenius ||
            throw(ArgumentError("KineticMonteCarloLoop rate_rule must be :metropolis or :arrhenius, got $(repr(RateRule))."))
        return new{RateRule,T}(r0, max(0, full_refresh_interval))
    end
end

function KineticMonteCarloLoop(; r0 = 1.0, full_refresh_interval = 0, rate_rule::Symbol = :metropolis)
    if rate_rule === :metropolis
        return KineticMonteCarloLoop{:metropolis}(; r0, full_refresh_interval)
    elseif rate_rule === :arrhenius
        return KineticMonteCarloLoop{:arrhenius}(; r0, full_refresh_interval)
    else
        throw(ArgumentError("KineticMonteCarloLoop rate_rule must be :metropolis or :arrhenius, got $(repr(rate_rule))."))
    end
end

KineticMonteCarloLoop{RateRule}(; r0 = 1.0, full_refresh_interval = 0) where {RateRule} =
    KineticMonteCarloLoop{RateRule,typeof(r0)}(r0, Int(full_refresh_interval))

@inline IsingKineticMonteCarloLoop(; kwargs...) = Unique(KineticMonteCarloLoop(; kwargs...))

mutable struct _KMCFenwick{T}
    tree::Vector{T}
end

@inline _KMCFenwick(::Type{T}, n::Int) where {T} = _KMCFenwick(zeros(T, n))

@inline function _kmc_fenwick_add!(ft::_KMCFenwick{T}, i::Int, delta::T) where {T}
    n = length(ft.tree)
    while i <= n
        @inbounds ft.tree[i] += delta
        i += i & -i
    end
    return ft
end

@inline function _kmc_fenwick_sum(ft::_KMCFenwick{T}, i::Int) where {T}
    s = zero(T)
    while i > 0
        @inbounds s += ft.tree[i]
        i -= i & -i
    end
    return s
end

@inline _kmc_total_rate(ft::_KMCFenwick) = _kmc_fenwick_sum(ft, length(ft.tree))

@inline function _kmc_fenwick_find(ft::_KMCFenwick{T}, u::T) where {T}
    n = length(ft.tree)
    n == 0 && return 0

    idx = 0
    s = zero(T)
    mask = prevpow(2, n)
    while mask != 0
        candidate = idx + mask
        if candidate <= n
            v = @inbounds ft.tree[candidate]
            if s + v < u
                s += v
                idx = candidate
            end
        end
        mask >>= 1
    end
    return min(idx + 1, n)
end

mutable struct _KMCEventTable{T}
    active_spins::Vector{Int}
    active_slots::Vector{Int}
    ΔEs::Vector{T}
    rates::Vector{T}
    targets::Vector{T}
    layer_idxs::Vector{Int}
    fenwick::_KMCFenwick{T}
    totalrate::T
    r0::T
end

@inline Base.eltype(::_KMCEventTable{T}) where {T} = T

function _KMCEventTable(model, active_spins, r0::T) where {T}
    active = collect(Int, active_spins)
    active_slots = zeros(Int, InteractiveIsing.nstates(model))
    @inbounds for slot in eachindex(active)
        active_slots[active[slot]] = slot
    end
    n = length(active)
    return _KMCEventTable(
        active,
        active_slots,
        zeros(T, n),
        zeros(T, n),
        Vector{T}(undef, n),
        zeros(Int, n),
        _KMCFenwick(T, n),
        zero(T),
        r0,
    )
end

@inline function _kmc_active_spin_vector(index_source)
    active_spins = sampling_indices(index_source)
    return active_spins isa AbstractVector ? active_spins : collect(active_spins)
end

function _kmc_set_active_spins!(events::_KMCEventTable{T}, model, active_spins) where {T}
    active = events.active_spins
    resize!(active, length(active_spins))
    @inbounds for i in eachindex(active_spins)
        active[i] = active_spins[i]
    end

    nstates_model = InteractiveIsing.nstates(model)
    resize!(events.active_slots, nstates_model)
    fill!(events.active_slots, 0)
    @inbounds for slot in eachindex(active)
        events.active_slots[active[slot]] = slot
    end

    n = length(active)
    resize!(events.ΔEs, n)
    resize!(events.rates, n)
    resize!(events.targets, n)
    resize!(events.layer_idxs, n)
    resize!(events.fenwick.tree, n)
    fill!(events.ΔEs, zero(T))
    fill!(events.rates, zero(T))
    fill!(events.fenwick.tree, zero(T))
    events.totalrate = zero(T)
    return events
end

@inline function _kmc_proposal_for_spin(proposer::LocalProposer, rng, spin_idx::Int)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[spin_idx]::statetype(proposer)
    layer_idx = spin_idx_to_layer_idx(spin_idx, proposer.layers)
    newstate = @inline inline_layer_dispatch(
        layer -> (@inline localrandstate(rng, layer, oldstate, proposer.delta)),
        layer_idx,
        proposer.layers,
    )
    return FlipProposal{statetype(proposer)}(spin_idx, oldstate, newstate, layer_idx, false)
end

@inline function _kmc_proposal_for_spin(proposer, rng, spin_idx::Int)
    spins = @inline InteractiveIsing.state(proposer.state)
    oldstate = @inbounds spins[spin_idx]::statetype(proposer)
    layer_idx = spin_idx_to_layer_idx(spin_idx, proposer.layers)
    newstate = @inline inline_layer_dispatch(
        layer -> (@inline randstate(rng, layer, oldstate)),
        layer_idx,
        proposer.layers,
    )
    return FlipProposal{statetype(proposer)}(spin_idx, oldstate, newstate, layer_idx, false)
end

@inline function _kmc_rate(::KineticMonteCarloLoop{:metropolis}, r0::T, ΔE::T, t::T) where {T}
    if !(t > zero(T))
        return ΔE <= zero(T) ? r0 : zero(T)
    end
    ΔE <= zero(T) && return r0
    exponent = clamp(-ΔE / t, T(-80), zero(T))
    rate = r0 * exp(exponent)
    return isfinite(rate) && rate >= zero(T) ? rate : zero(T)
end

@inline function _kmc_rate(::KineticMonteCarloLoop{:arrhenius}, r0::T, ΔE::T, t::T) where {T}
    if !(t > zero(T))
        return ΔE <= zero(T) ? r0 : zero(T)
    end
    exponent = clamp(-ΔE / t, T(-80), T(80))
    rate = r0 * exp(exponent)
    return isfinite(rate) && rate >= zero(T) ? rate : zero(T)
end

@inline function _kmc_refresh_slot!(events::_KMCEventTable{T}, kinetic, context, slot::Int, t::T) where {T}
    (;model, hamiltonian, proposer, rng) = context
    spin_idx = @inbounds events.active_spins[slot]
    proposal = @inline _kmc_proposal_for_spin(proposer, rng, spin_idx)
    ΔE = T(@inline calculate(ΔH(), hamiltonian, model, proposal))
    rate = @inline _kmc_rate(kinetic, events.r0, ΔE, t)

    oldrate = @inbounds events.rates[slot]
    delta_rate = rate - oldrate
    @inbounds begin
        events.ΔEs[slot] = ΔE
        events.rates[slot] = rate
        events.targets[slot] = T(to_val(proposal))
        events.layer_idxs[slot] = proposal.layer_idx
    end
    @inline _kmc_fenwick_add!(events.fenwick, slot, delta_rate)
    events.totalrate += delta_rate
    return events
end

function _kmc_rebuild_rates!(events::_KMCEventTable{T}, kinetic, context, t::T) where {T}
    fill!(events.fenwick.tree, zero(T))
    events.totalrate = zero(T)
    @inbounds for slot in eachindex(events.active_spins)
        proposal = @inline _kmc_proposal_for_spin(context.proposer, context.rng, events.active_spins[slot])
        ΔE = T(@inline calculate(ΔH(), context.hamiltonian, context.model, proposal))
        rate = @inline _kmc_rate(kinetic, events.r0, ΔE, t)
        events.ΔEs[slot] = ΔE
        events.rates[slot] = rate
        events.targets[slot] = T(to_val(proposal))
        events.layer_idxs[slot] = proposal.layer_idx
        events.totalrate += rate
    end

    copyto!(events.fenwick.tree, events.rates)
    tree = events.fenwick.tree
    @inbounds for i in eachindex(tree)
        j = i + (i & -i)
        if j <= length(tree)
            tree[j] += tree[i]
        end
    end
    return events
end

@inline function _kmc_refresh_affected!(events::_KMCEventTable{T}, kinetic, context, spin_idx::Int, t::T) where {T}
    slot = @inbounds events.active_slots[spin_idx]
    slot > 0 && (@inline _kmc_refresh_slot!(events, kinetic, context, slot, t))

    rowvals = SparseArrays.rowvals(context.adj)
    for ptr in SparseArrays.nzrange(context.adj, spin_idx)
        neighbour = @inbounds rowvals[ptr]
        neighbour == spin_idx && continue
        neighbour_slot = @inbounds events.active_slots[neighbour]
        neighbour_slot > 0 && (@inline _kmc_refresh_slot!(events, kinetic, context, neighbour_slot, t))
    end
    return events
end

@inline function _kmc_draw_event(rng, events::_KMCEventTable{T}) where {T}
    totalrate = events.totalrate
    if !(totalrate > zero(T)) || !isfinite(totalrate)
        return 0, zero(T)
    end
    u = max(rand(rng, T) * totalrate, eps(T))
    u = min(u, totalrate)
    return _kmc_fenwick_find(events.fenwick, u), totalrate
end

@inline update!(::KineticMonteCarloLoop, hterm::HamiltonianTerms, model::AbstractIsingGraph, proposal::FlipProposal) =
    update!(Metropolis(), hterm, model, proposal)

@inline function Processes.init(kinetic::KineticMonteCarloLoop, context::Cont) where {Cont}
    (;model) = context

    active_index_set = index_set(model)
    active_spins = collect(@inline _kmc_active_spin_vector(active_index_set))
    adj = InteractiveIsing.adj(model)
    hamiltonian = init!(model.hamiltonian, model)
    proposer = get_proposer(model)
    rng = Random.MersenneTwister()

    SType = eltype(model)
    t = SType(temp(model))
    events = _KMCEventTable(model, active_spins, SType(kinetic.r0))
    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
    ΔE = zero(SType)
    dt = zero(SType)
    totalrate = zero(SType)
    event_index = 0
    spin_idx = 0
    time = Ref(zero(SType))
    lasttemp = Ref(t)
    steps_since_refresh = Ref(0)

    init_context = (;model, active_index_set, adj, hamiltonian, proposer, rng)
    @inline _kmc_rebuild_rates!(events, kinetic, init_context, t)

    return (;init_context..., events, proposal, ΔE, dt, totalrate, event_index,
        spin_idx, time, lasttemp, steps_since_refresh, T = t)
end

@inline function Processes.step!(kinetic::KineticMonteCarloLoop, context::C) where {C}
    (;model, rng, proposer, events) = context
    SType = eltype(model)
    t = SType(temp(model))

    active_changed = @inline consume_changed!(context.active_index_set)
    temp_changed = !isfinite(context.lasttemp[]) ||
        abs(t - context.lasttemp[]) > eps(SType) * max(abs(t), abs(context.lasttemp[]), one(SType))
    refresh_due = kinetic.full_refresh_interval > 0 &&
        context.steps_since_refresh[] >= kinetic.full_refresh_interval

    if active_changed
        @inline _kmc_set_active_spins!(events, model, @inline _kmc_active_spin_vector(context.active_index_set))
        @inline _kmc_rebuild_rates!(events, kinetic, context, t)
        context.steps_since_refresh[] = 0
    elseif temp_changed || refresh_due
        @inline _kmc_rebuild_rates!(events, kinetic, context, t)
        context.steps_since_refresh[] = 0
    end
    context.lasttemp[] = t

    event_index, totalrate = @inline _kmc_draw_event(rng, events)
    if event_index == 0
        dt = zero(SType)
        proposal = context.proposal
        yield()
        return (;proposal, ΔE = zero(SType), dt, totalrate = zero(SType),
            event_index, spin_idx = 0, kmc_time = context.time[], T = t)
    end

    spin_idx = @inbounds events.active_spins[event_index]
    spins = @inline InteractiveIsing.state(model)
    oldstate = @inbounds spins[spin_idx]
    proposal = FlipProposal{SType}(
        spin_idx,
        oldstate,
        @inbounds(events.targets[event_index]),
        @inbounds(events.layer_idxs[event_index]),
        false,
    )
    ΔE = @inbounds events.ΔEs[event_index]

    proposal = @inline accept(proposer, proposal)
    @inline update!(kinetic, context.hamiltonian, model, proposal)

    dt = -log(max(rand(rng, SType), eps(SType))) / totalrate
    context.time[] += dt
    context.steps_since_refresh[] += 1

    @inline _kmc_refresh_affected!(events, kinetic, context, spin_idx, t)

    return (;proposal, ΔE, dt, totalrate, event_index, spin_idx,
        kmc_time = context.time[], T = t)
end
