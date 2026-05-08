export GlobalLangevin

"""
    GlobalLangevin(; stepsize=0.1, max_drift_fraction=0.15, group_steps=1, adjusted=false)

Global-gradient Langevin Monte Carlo update.

At the start of each internal cycle, one `Processes.step!` refreshes the
derivative for all active spins at the current state. That step and subsequent
steps then consume the cached derivatives one spin at a time. The default
`adjusted=false` reflects proposals back into the state bounds and accepts them
directly, which is practical for large bounded systems but not an exact
Boltzmann sampler.

`stepsize` is the proposal size used by each single-spin Langevin proposal.
`group_steps` is retained for interface compatibility; a `Processes.step!` call
still attempts only one spin update.

With `adjusted=true`, the selected spin move is accepted or rejected with a
Metropolis-adjusted Langevin acceptance probability.
"""
struct GlobalLangevin{T<:Real} <: IsingMCAlgorithm
    stepsize::T
    max_drift_fraction::T
    group_steps::Int
    adjusted::Bool
end

function GlobalLangevin(; stepsize = 0.1, max_drift_fraction = 0.15, group_steps = 1, adjusted = false)
    stepsize, max_drift_fraction = promote(stepsize, max_drift_fraction)
    return GlobalLangevin(stepsize, max_drift_fraction, max(1, Int(group_steps)), adjusted)
end

GlobalLangevin(adjusted::Bool) = GlobalLangevin(; adjusted)

@inline function Processes.init(langevin::GlobalLangevin, context::Cont) where {Cont}
    (;model) = context

    for layer in layers(model)
        statetype(layer) isa Discrete &&
            error("GlobalLangevin requires Continuous layers; layer $(layeridx(layer)) is Discrete. " *
                  "Use a Metropolis or Heatbath algorithm for discrete spin models.")
    end

    hamiltonian = model.hamiltonian
    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    nstates_model = InteractiveIsing.nstates(model)
    dH_prealloc = zeros(eltype(model), nstates_model)
    active_index_set = index_set(model)
    active_spins = collect(@inline _active_spin_vector(active_index_set))
    layer_views = layers(model)
    T = temp(model)
    SType = eltype(model)
    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(_langevin_unwrap_ref(_langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    group_steps = Ref(langevin.group_steps)
    adjusted = Ref(langevin.adjusted)

    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
    ΔE = zero(SType)
    accepted = 0
    attempted = 0
    gradient_max = zero(SType)
    gradient_rms = zero(SType)
    reflected_fraction = zero(SType)
    schedule_idxs = Vector{Int}(undef, length(active_spins))
    schedule_position = Ref(0)
    schedule_length = Ref(0)
    gradient_max_cache = Ref(zero(SType))
    gradient_sumsq_cache = Ref(zero(SType))

    η = max(stepsize[], eps(SType))
    σ = zero(SType)

    return (;model, hamiltonian, rng, dH_prealloc, active_index_set, active_spins,
                layer_views, stepsize, max_drift_fraction, group_steps,
                adjusted, proposal, ΔE, accepted, attempted, T, η, σ,
                gradient_max, gradient_rms, reflected_fraction,
                schedule_idxs, schedule_position, schedule_length,
                gradient_max_cache, gradient_sumsq_cache)
end

@inline update!(::GlobalLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function _langevin_accept_single_spin!(
    langevin,
    hamiltonian,
    model,
    spin_idx::Int,
    layer_idx::Int,
    old_state::T,
    new_state::T,
) where {T}
    spins = @inline InteractiveIsing.graphstate(model)
    proposal = FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, true)
    @inbounds spins[spin_idx] = new_state
    @inline update!(langevin, hamiltonian, model, proposal)
    return proposal
end

@inline function _langevin_single_spin_proposal!(
    langevin,
    rng,
    dh,
    hamiltonian,
    model,
    layer_views,
    spin_idx::Int,
    derivative::T,
    η::T,
    σ::T,
    drift_fraction::T,
    t::T,
    use_adjusted::Bool,
) where {T}
    spins = @inline InteractiveIsing.graphstate(model)
    low_state, high_state, local_span, layer_idx = @inline _local_langevin_bounds(spin_idx, layer_views)
    local_drift_cap = drift_fraction * local_span
    drift_step = use_adjusted ? η * derivative : (@inline _langevin_drift_step(η, derivative, local_drift_cap))

    old_state = @inbounds spins[spin_idx]
    noise = σ > zero(T) ? (@inline randn(rng, T)) : zero(T)
    trial_state = old_state - drift_step + σ * noise
    reflected = 0
    ΔE = zero(T)

    if !use_adjusted
        new_state = @inline _reflect_to_bounds(trial_state, low_state, high_state)
        reflected = new_state == trial_state ? 0 : 1
        if t <= zero(T)
            new_state, _ = @inline _langevin_zero_temp_relax_state(
                dh,
                hamiltonian,
                model,
                spin_idx,
                old_state,
                new_state,
                derivative,
            )
        end
        proposal = @inline _langevin_accept_single_spin!(
            langevin,
            hamiltonian,
            model,
            spin_idx,
            layer_idx,
            old_state,
            new_state,
        )
        return proposal, ΔE, 1, reflected
    end

    if !(@inline _in_bounds(trial_state, low_state, high_state))
        proposal = FlipProposal{T}(spin_idx, old_state, old_state, layer_idx, false)
        return proposal, ΔE, 0, reflected
    end

    new_state = trial_state
    proposal_trial = FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, false)
    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal_trial)

    if t <= zero(T)
        accept_move = isfinite(ΔE) && ΔE <= zero(T)
    else
        four_ηT = T(4) * η * max(t, eps(T))
        forward_mean = old_state - drift_step
        reverse_mean = @inline _local_langevin_reverse_mean(dh, hamiltonian, model, spin_idx, old_state, new_state, η)
        log_forward_q = @inline _mala_log_kernel(new_state, forward_mean, four_ηT)
        log_reverse_q = @inline _mala_log_kernel(old_state, reverse_mean, four_ηT)
        log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
        accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(T) || log(@inline rand(rng, T)) < log_acceptance)
    end

    if accept_move
        proposal = @inline _langevin_accept_single_spin!(
            langevin,
            hamiltonian,
            model,
            spin_idx,
            layer_idx,
            old_state,
            new_state,
        )
        return proposal, ΔE, 1, reflected
    end

    proposal = FlipProposal{T}(spin_idx, old_state, new_state, layer_idx, false)
    return proposal, ΔE, 0, reflected
end

@inline function Processes.step!(langevin::GlobalLangevin, context::C) where {C}
    (;hamiltonian, rng, model, dH_prealloc, layer_views, stepsize,
        max_drift_fraction, group_steps, adjusted, schedule_idxs,
        schedule_position, schedule_length, gradient_max_cache,
        gradient_sumsq_cache) = context

    SType = eltype(model)
    epsT = eps(SType)
    T = temp(model)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    n_group_steps = max(1, group_steps[])
    use_adjusted = adjusted[]
    dh = d_iH()
    active_changed = @inline consume_changed!(context.active_index_set)
    if active_changed
        @inline _set_local_langevin_active_spins!(context.active_spins, @inline _active_spin_vector(context.active_index_set))
    end
    active_spins = context.active_spins
    n = length(active_spins)
    if n == 0
        schedule_position[] = 0
        schedule_length[] = 0
        σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
        proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
        return (;proposal, ΔE = zero(SType), accepted = 0, attempted = 0,
            acceptance_rate = zero(SType), T, η, σ, group_steps = n_group_steps,
            refreshed_gradient = false, gradient_max = zero(SType),
            gradient_rms = zero(SType), reflected_fraction = zero(SType))
    end

    if use_adjusted
        gradient_max = zero(SType)
        gradient_sumsq = zero(SType)
        @inbounds for spin_idx in active_spins
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = @inline _finite_derivative(SType(derivative))
            dH_prealloc[spin_idx] = derivative
            gradient_sumsq += derivative * derivative
            gradient_max = max(gradient_max, abs(derivative))
        end

        k = @inline rand(rng, 1:n)
        spin_idx = @inbounds active_spins[k]
        derivative = @inbounds dH_prealloc[spin_idx]
        σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
        proposal, ΔE, accepted, reflected = @inline _langevin_single_spin_proposal!(
            langevin,
            rng,
            dh,
            hamiltonian,
            model,
            layer_views,
            spin_idx,
            derivative,
            η,
            σ,
            drift_fraction,
            t,
            true,
        )

        attempted = 1
        acceptance_rate = SType(accepted)
        gradient_rms = sqrt(gradient_sumsq / SType(n))
        reflected_fraction = SType(reflected)
        schedule_position[] = 0
        schedule_length[] = 0
        return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
            group_steps = n_group_steps, refreshed_gradient = true,
            gradient_max, gradient_rms, reflected_fraction)
    end

    refreshed = active_changed || schedule_position[] == 0 || schedule_position[] > schedule_length[]
    if refreshed
        resize!(schedule_idxs, n)
        gradient_max = zero(SType)
        gradient_sumsq = zero(SType)
        @inbounds for pos in 1:n
            spin_idx = active_spins[pos]
            schedule_idxs[pos] = spin_idx
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = @inline _finite_derivative(SType(derivative))
            dH_prealloc[spin_idx] = derivative
            gradient_sumsq += derivative * derivative
            gradient_max = max(gradient_max, abs(derivative))
        end
        @inbounds for pos in 1:n
            swap_pos = @inline rand(rng, pos:n)
            schedule_idxs[pos], schedule_idxs[swap_pos] = schedule_idxs[swap_pos], schedule_idxs[pos]
        end
        schedule_position[] = 1
        schedule_length[] = n
        gradient_max_cache[] = gradient_max
        gradient_sumsq_cache[] = gradient_sumsq
    end

    pos = schedule_position[]
    spin_idx = @inbounds schedule_idxs[pos]
    derivative = @inbounds dH_prealloc[spin_idx]
    σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
    proposal, ΔE, accepted, reflected = @inline _langevin_single_spin_proposal!(
        langevin,
        rng,
        dh,
        hamiltonian,
        model,
        layer_views,
        spin_idx,
        derivative,
        η,
        σ,
        drift_fraction,
        t,
        use_adjusted,
    )
    schedule_position[] = pos + 1

    attempted = 1
    acceptance_rate = attempted == 0 ? zero(SType) : SType(accepted) / SType(attempted)
    gradient_max = gradient_max_cache[]
    gradient_rms = schedule_length[] == 0 ? zero(SType) : sqrt(gradient_sumsq_cache[] / SType(schedule_length[]))
    reflected_fraction = SType(reflected)
    return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
        group_steps = n_group_steps, refreshed_gradient = refreshed,
        gradient_max, gradient_rms, reflected_fraction)
end
