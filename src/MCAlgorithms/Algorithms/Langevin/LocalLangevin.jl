export LocalLangevin

"""
    LocalLangevin(; stepsize=0.1, max_drift_fraction=0.15, group_steps=1, adjusted=true, order=:random)

Single-spin Langevin Monte Carlo update.

Each `StatefulAlgorithms.step!` attempts one continuous spin move using the local energy
derivative. The selected spin is taken from an ordered sweep over the active
spins. `adjusted` and `order` are type parameters, so the inner step specializes
on the chosen proposal structure. With `adjusted=true`, each proposal uses a
Metropolis-adjusted Langevin acceptance probability (MALA-style) for Boltzmann
correctness of the discretized proposal. With `adjusted=false`, moves are always
accepted after drift limiting, deterministic boundary clamping, and stochastic
boundary reflection, which is faster but not an exact Boltzmann sampler.

`stepsize` is the proposal size used by each single-spin Langevin proposal.
`group_steps` is the number of full sweeps in one internal cycle; it does not
divide or reinterpret `stepsize`.
`order=:cyclic` uses the previous fast random-offset cyclic sweep. `order=:random`
uses a lazy random permutation each sweep. `order=:deterministic` uses the active
spin order directly.
"""
struct LocalLangevin{Order,Adjusted,T<:Real} <: IsingMCAlgorithm
    stepsize::T
    max_drift_fraction::T
    group_steps::Int

    function LocalLangevin{Order,Adjusted,T}(
        stepsize::T,
        max_drift_fraction::T,
        group_steps::Int,
    ) where {Order,Adjusted,T<:Real}
        Order === :cyclic || Order === :random || Order === :deterministic ||
            throw(ArgumentError("LocalLangevin order must be :cyclic, :random, or :deterministic, got $(repr(Order))."))
        Adjusted === true || Adjusted === false ||
            throw(ArgumentError("LocalLangevin adjusted must be true or false, got $(repr(Adjusted))."))
        return new{Order,Adjusted,T}(stepsize, max_drift_fraction, max(1, group_steps))
    end
end

function LocalLangevin(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    group_steps = 1,
    adjusted::Bool = true,
    order::Symbol = :random,
)
    if order === :cyclic
        return LocalLangevin{:cyclic,adjusted}(; stepsize, max_drift_fraction, group_steps)
    elseif order === :random
        return LocalLangevin{:random,adjusted}(; stepsize, max_drift_fraction, group_steps)
    elseif order === :deterministic
        return LocalLangevin{:deterministic,adjusted}(; stepsize, max_drift_fraction, group_steps)
    else
        throw(ArgumentError("LocalLangevin order must be :cyclic, :random, or :deterministic, got $(repr(order))."))
    end
end

function LocalLangevin{Order}(; stepsize = 0.1, max_drift_fraction = 0.15, group_steps = 1, adjusted::Bool = true) where {Order}
    return LocalLangevin{Order,adjusted}(; stepsize, max_drift_fraction, group_steps)
end

function LocalLangevin{Order,Adjusted}(; stepsize = 0.1, max_drift_fraction = 0.15, group_steps = 1) where {Order,Adjusted}
    stepsize, max_drift_fraction = promote(stepsize, max_drift_fraction)
    return LocalLangevin{Order,Adjusted,typeof(stepsize)}(stepsize, max_drift_fraction, Int(group_steps))
end

LocalLangevin(stepsize::Real, max_drift_fraction::Real; group_steps = 1, adjusted::Bool = true, order::Symbol = :random) =
    LocalLangevin(; stepsize, max_drift_fraction, group_steps, adjusted, order)

LocalLangevin(adjusted::Bool) = LocalLangevin(; adjusted)

@inline function IsingLangevin()
    return Unique(LocalLangevin())
end

@inline _in_bounds(x, lo, hi) = isfinite(x) && lo <= x <= hi

"""
    _reflect_to_bounds(x, lo, hi)

Reflect `x` back into the closed interval `[lo, hi]`.

This is used by the unadjusted Langevin path to avoid hard clamping at the
boundary.
"""
@inline function _reflect_to_bounds(x::T, lo::T, hi::T) where {T}
    if !isfinite(x)
        return lo
    end
    if lo <= x <= hi
        return x
    end

    span = hi - lo
    if !(span > zero(T))
        return lo
    end

    y = mod(x - lo, T(2) * span)
    return y <= span ? lo + y : hi - (y - span)
end

@inline function _finite_derivative(x::T) where {T}
    return isfinite(x) ? x : zero(T)
end

@inline function _langevin_drift_step(η, derivative, cap)
    return clamp(η * derivative, -cap, cap)
end

@inline function _langevin_boundary_drift_step(drift_step::T, old_state::T, low_state::T, high_state::T) where {T}
    if old_state <= low_state && drift_step > zero(T)
        return zero(T)
    elseif old_state >= high_state && drift_step < zero(T)
        return zero(T)
    end
    return drift_step
end

@inline function _langevin_unadjusted_state(
    old_state::T,
    drift_step::T,
    noise_step::T,
    low_state::T,
    high_state::T,
) where {T}
    drifted_state = clamp(old_state - drift_step, low_state, high_state)
    noise_trial = drifted_state + noise_step
    new_state = @inline _reflect_to_bounds(noise_trial, low_state, high_state)
    reflected = new_state == noise_trial ? 0 : 1
    return new_state, reflected
end

@inline function _mala_log_kernel(x, mean, four_ηT)
    dx = x - mean
    return -(dx * dx) / four_ηT
end

@inline function _langevin_derivative_sign_flipped(a, b)
    return (a > zero(a) && b < zero(b)) || (a < zero(a) && b > zero(b))
end

@inline function _langevin_zero_temp_relax_state(
    dh,
    hamiltonian,
    model,
    spin_idx::Int,
    old_state::T,
    trial_state::T,
    derivative::T,
) where {T}
    iszero(derivative) && return old_state, derivative

    spins = @inline InteractiveIsing.graphstate(model)
    candidate = trial_state
    candidate_derivative = derivative

    for _ in 1:24
        @inbounds spins[spin_idx] = candidate
        candidate_derivative = @inline _finite_derivative(T(@inline calculate(dh, hamiltonian, model, spin_idx)))
        @inbounds spins[spin_idx] = old_state

        if isfinite(candidate_derivative) && !(@inline _langevin_derivative_sign_flipped(derivative, candidate_derivative))
            return candidate, candidate_derivative
        end

        candidate = (old_state + candidate) / T(2)
        candidate == old_state && return old_state, derivative
    end

    return candidate, candidate_derivative
end

"""
    _active_spin_vector(model)

Return the model's active sampling indices as an `AbstractVector`.

Langevin algorithms need random access into the active index set, so non-vector
iterables are materialized here.
"""
@inline function _active_spin_vector(model)
    active_spins = sampling_indices(model)
    return active_spins isa AbstractVector ? active_spins : collect(active_spins)
end

@inline _langevin_context_value(context, name::Symbol, default) = get(context, name, default)
@inline _langevin_unwrap_ref(x) = x isa Ref ? x[] : x

@inline function StatefulAlgorithms.init(langevin::LocalLangevin{Order,Adjusted}, context::Cont) where {Order,Adjusted,Cont}
    (;model) = context

    active_index_set = index_set(model)
    active_spins = collect(@inline _active_spin_vector(active_index_set))
    layer_views = layers(model)
    for spin_idx in active_spins
        is_discrete = @inline spin_idx_layer_dispatch(layer -> statetype(layer) isa Discrete, spin_idx, layer_views)
        is_discrete &&
            error("LocalLangevin requires Continuous active layers; layer $(spin_idx_to_layer_idx(spin_idx, layer_views)) is Discrete. " *
                  "Use a Metropolis or Heatbath algorithm for discrete spin models.")
    end

    hamiltonian = model.hamiltonian
    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    dH_prealloc = zeros(eltype(model), InteractiveIsing.nstates(model))
    SType = eltype(model)
    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(@inline _langevin_unwrap_ref(@inline _langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    group_steps_ref = Ref(langevin.group_steps)
    adjusted = Adjusted
    sweep_position = Ref(0)
    sweep_order = Order === :random ? collect(1:length(active_spins)) : Int[]
    sweep_offset = Ref(0)
    group_remaining = Ref(0)
    T = SType(temp(model))

    return (;model, hamiltonian, rng, dH_prealloc, active_index_set, active_spins,
                layer_views, stepsize, max_drift_fraction, group_steps_ref,
                adjusted, sweep_position, sweep_order, sweep_offset, group_remaining, T)
end

@inline update!(::LocalLangevin, hterm, model::AbstractIsingGraph, proposal::FlipProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function _reset_local_langevin_order!(order::Vector{Int}, n::Int)
    resize!(order, n)
    @inbounds for i in 1:n
        order[i] = i
    end
    return order
end

@inline function _set_local_langevin_active_spins!(cached::Vector{Int}, current)
    resize!(cached, length(current))
    @inbounds for i in eachindex(current)
        cached[i] = current[i]
    end
    return cached
end

@inline function _start_local_langevin_cycle!(langevin::LocalLangevin{Order}, context, dh, hamiltonian, model, active_spins) where {Order}
    SType = eltype(model)
    gradient_max = zero(SType)
    for spin_idx in active_spins
        derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
        derivative = @inline _finite_derivative(SType(derivative))
        @inbounds context.dH_prealloc[spin_idx] = derivative
        gradient_max = max(gradient_max, abs(derivative))
    end

    n = length(active_spins)
    context.group_remaining[] = max(1, context.group_steps_ref[])
    context.sweep_position[] = 1
    if Order === :random
        @inline _reset_local_langevin_order!(context.sweep_order, n)
    end
    return gradient_max
end

@inline function _next_local_langevin_order_index!(langevin::LocalLangevin{Order}, context, rng, n::Int) where {Order}
    if Order === :random
        pos = context.sweep_position[]
        swap_pos = @inline rand(rng, pos:n)
        order = context.sweep_order
        @inbounds begin
            order[pos], order[swap_pos] = order[swap_pos], order[pos]
            return order[pos]
        end
    elseif Order === :cyclic
        pos = context.sweep_position[]
        if pos == 1
            context.sweep_offset[] = @inline rand(rng, 0:(n - 1))
        end
        k = pos + context.sweep_offset[]
        return k > n ? k - n : k
    elseif Order === :deterministic
        return context.sweep_position[]
    else
        throw(ArgumentError("LocalLangevin order must be :cyclic, :random, or :deterministic, got $(repr(Order))."))
    end
end

@inline function _advance_local_langevin_cursor!(langevin::LocalLangevin{Order}, context, n::Int) where {Order}
    context.sweep_position[] += 1
    if context.sweep_position[] <= n
        return nothing
    end

    context.group_remaining[] -= 1
    if context.group_remaining[] > 0
        context.sweep_position[] = 1
        if Order === :random
            @inline _reset_local_langevin_order!(context.sweep_order, n)
        end
    else
        context.sweep_position[] = 0
    end
    return nothing
end

@inline function _local_langevin_bounds(spin_idx::Int, layer_views)
    local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, layer_views)
    low_state = local_states[1]
    high_state = local_states[end]
    layer_idx = @inline spin_idx_to_layer_idx(spin_idx, layer_views)
    return low_state, high_state, high_state - low_state, layer_idx
end

@inline function _local_langevin_derivative!(dH_prealloc, dh, hamiltonian, model, spin_idx::Int)
    SType = eltype(model)
    derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
    derivative = @inline _finite_derivative(SType(derivative))
    @inbounds dH_prealloc[spin_idx] = derivative
    return derivative
end

"""
    _local_langevin_accept!(...)

Commit one local spin proposal, update Hamiltonian caches, and refresh the
stored local derivative at that spin.
"""
@inline function _local_langevin_accept!(
    langevin::LocalLangevin,
    dh,
    hamiltonian,
    model,
    dH_prealloc,
    spin_idx::Int,
    layer_idx::Int,
    old_state::T,
    new_state::T,
) where {T}
    spins = @inline InteractiveIsing.graphstate(model)
    proposal = FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, true)
    @inbounds spins[spin_idx] = new_state
    @inline update!(langevin, hamiltonian, model, proposal)
    post_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
    post_derivative = @inline _finite_derivative(T(post_derivative))
    @inbounds dH_prealloc[spin_idx] = post_derivative
    return proposal, post_derivative
end

"""
    _local_langevin_unadjusted!(...)

Apply the fast always-accepted local proposal. It clamps deterministic drift into
the local state bounds, reflects only the stochastic displacement, and at zero
temperature backs off a crossing move that would otherwise bounce around the
local stationary point.
"""
@inline function _local_langevin_unadjusted!(
    langevin::LocalLangevin,
    dh,
    hamiltonian,
    model,
    dH_prealloc,
    spin_idx::Int,
    layer_idx::Int,
    old_state::T,
    drift_step::T,
    noise_step::T,
    low_state::T,
    high_state::T,
    derivative::T,
    t::T,
) where {T}
    new_state, reflected = @inline _langevin_unadjusted_state(
        old_state,
        drift_step,
        noise_step,
        low_state,
        high_state,
    )

    if t <= zero(T)
        post_derivative = derivative
        new_state, post_derivative = @inline _langevin_zero_temp_relax_state(
            dh,
            hamiltonian,
            model,
            spin_idx,
            old_state,
            new_state,
            derivative,
        )
        spins = @inline InteractiveIsing.graphstate(model)
        proposal = FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, true)
        @inbounds spins[spin_idx] = new_state
        @inline update!(langevin, hamiltonian, model, proposal)
        @inbounds dH_prealloc[spin_idx] = post_derivative
        return proposal, post_derivative, reflected
    end

    proposal, post_derivative = @inline _local_langevin_accept!(
        langevin,
        dh,
        hamiltonian,
        model,
        dH_prealloc,
        spin_idx,
        layer_idx,
        old_state,
        new_state,
    )
    return proposal, post_derivative, reflected
end

"""
    _local_langevin_reverse_mean(...)

Temporarily evaluate the local derivative at the trial state and return the
reverse Langevin proposal mean used in the MALA correction.
"""
@inline function _local_langevin_reverse_mean(
    dh,
    hamiltonian,
    model,
    spin_idx::Int,
    old_state::T,
    new_state::T,
    η::T,
) where {T}
    spins = @inline InteractiveIsing.graphstate(model)
    @inbounds spins[spin_idx] = new_state
    reverse_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
    reverse_derivative = @inline _finite_derivative(T(reverse_derivative))
    reverse_mean = new_state - η * reverse_derivative
    @inbounds spins[spin_idx] = old_state
    return reverse_mean
end

"""
    _local_langevin_adjusted!(...)

Evaluate one Metropolis-adjusted local Langevin proposal. Accepted moves are
committed through `_local_langevin_accept!`; rejected moves leave the graph and
Hamiltonian caches unchanged.
"""
@inline function _local_langevin_adjusted!(
    langevin::LocalLangevin,
    rng,
    dh,
    hamiltonian,
    model,
    dH_prealloc,
    spin_idx::Int,
    layer_idx::Int,
    old_state::T,
    trial_state::T,
    low_state::T,
    high_state::T,
    derivative::T,
    drift_step::T,
    t::T,
    η::T,
    four_ηT::T,
) where {T}
    if !(@inline _in_bounds(trial_state, low_state, high_state))
        return FlipProposal{T}(spin_idx, old_state, old_state, layer_idx, false), zero(T), 0, derivative
    end

    new_state = trial_state
    proposal_trial = FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, false)
    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal_trial)

    if t <= zero(T)
        accept_move = isfinite(ΔE) && ΔE <= zero(T)
    else
        forward_mean = old_state - drift_step
        reverse_mean = @inline _local_langevin_reverse_mean(dh, hamiltonian, model, spin_idx, old_state, new_state, η)
        log_forward_q = @inline _mala_log_kernel(new_state, forward_mean, four_ηT)
        log_reverse_q = @inline _mala_log_kernel(old_state, reverse_mean, four_ηT)
        log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
        accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(T) || log(@inline rand(rng, T)) < log_acceptance)
    end

    if accept_move
        proposal, post_derivative = @inline _local_langevin_accept!(
            langevin,
            dh,
            hamiltonian,
            model,
            dH_prealloc,
            spin_idx,
            layer_idx,
            old_state,
            new_state,
        )
        return proposal, ΔE, 1, post_derivative
    end

    return FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, false), ΔE, 0, derivative
end

@inline function StatefulAlgorithms.step!(langevin::LocalLangevin{Order,Adjusted}, context::C) where {Order,Adjusted,C}
    (;hamiltonian, rng, model, dH_prealloc, layer_views, stepsize,
        max_drift_fraction, T) = context

    SType = eltype(model)
    spins = @inline InteractiveIsing.graphstate(model)
    epsT = eps(SType)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    dh = d_iH()
    gradient_max = SType(@inline _langevin_unwrap_ref(@inline _langevin_context_value(context, :gradient_max, zero(SType))))
    active_index_set = @inline _langevin_context_value(context, :active_index_set, model)
    active_changed = @inline consume_changed!(active_index_set)
    if active_changed
        @inline _set_local_langevin_active_spins!(context.active_spins, @inline _active_spin_vector(active_index_set))
    end
    active_spins = context.active_spins

    σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
    n = length(active_spins)
    if n == 0
        context.sweep_position[] = 0
        context.group_remaining[] = 0
        return (;)
    end
    if active_changed || context.sweep_position[] == 0 || context.sweep_position[] > n
        gradient_max = @inline _start_local_langevin_cycle!(langevin, context, dh, hamiltonian, model, active_spins)
    end

    ΔE = zero(SType)
    four_ηT = SType(4) * η * max(t, epsT)
    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
    reflected = 0

    k = @inline _next_local_langevin_order_index!(langevin, context, rng, n)
    spin_idx = @inbounds active_spins[k]
    derivative = @inline _local_langevin_derivative!(dH_prealloc, dh, hamiltonian, model, spin_idx)
    gradient_max = max(gradient_max, abs(derivative))
    low_state, high_state, local_span, layer_idx = @inline _local_langevin_bounds(spin_idx, layer_views)

    local_drift_cap = drift_fraction * local_span
    raw_drift_step = η * derivative
    drift_step = Adjusted ? raw_drift_step : (@inline _langevin_drift_step(η, derivative, local_drift_cap))
    old_state = @inbounds spins[spin_idx]
    if !Adjusted
        drift_step = @inline _langevin_boundary_drift_step(drift_step, old_state, low_state, high_state)
    end

    noise = σ > zero(SType) ? (@inline randn(rng, SType)) : zero(SType)
    noise_step = σ * noise
    trial_state = old_state - drift_step + noise_step
    post_derivative = derivative
    accepted = 0

    if !Adjusted
        proposal, post_derivative, reflected = @inline _local_langevin_unadjusted!(
            langevin,
            dh,
            hamiltonian,
            model,
            dH_prealloc,
            spin_idx,
            layer_idx,
            old_state,
            drift_step,
            noise_step,
            low_state,
            high_state,
            derivative,
            t,
        )
        accepted += 1
    else
        proposal, ΔE, accepted, post_derivative = @inline _local_langevin_adjusted!(
            langevin,
            rng,
            dh,
            hamiltonian,
            model,
            dH_prealloc,
            spin_idx,
            layer_idx,
            old_state,
            trial_state,
            low_state,
            high_state,
            derivative,
            drift_step,
            t,
            η,
            four_ηT,
        )
    end
    gradient_max = max(gradient_max, abs(post_derivative))

    @inline _advance_local_langevin_cursor!(langevin, context, n)

    acceptance_rate = SType(accepted)
    gradient_rms = abs(derivative)
    reflected_fraction = SType(reflected)
   
    return (;proposal, ΔE, accepted, acceptance_rate, η, σ,
        gradient_max, gradient_rms, reflected_fraction)
    # return nothing
end
