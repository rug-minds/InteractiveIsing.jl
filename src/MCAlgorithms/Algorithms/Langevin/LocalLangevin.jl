export LocalLangevin

"""
    LocalLangevin(; stepsize=0.1, max_drift_fraction=0.15, group_steps=1, adjusted=true)

Single-spin Langevin Monte Carlo update.

Each step sweeps over the active sampling indices and proposes one continuous
spin move at a time using the local energy derivative. With `adjusted=true`,
each proposal uses a Metropolis-adjusted Langevin acceptance probability
(MALA-style) for Boltzmann correctness of the discretized proposal. With
`adjusted=false`, moves are always accepted after drift limiting and boundary
reflection, which is faster but not an exact Boltzmann sampler.

`stepsize` is the proposal size used by each single-spin Langevin proposal.
`group_steps` repeats that full local sweep several times inside one
`Processes.step!` call; it does not divide or reinterpret `stepsize`.
"""
struct LocalLangevin{T<:Real} <: IsingMCAlgorithm
    stepsize::T
    max_drift_fraction::T
    group_steps::Int
    adjusted::Bool
end

function LocalLangevin(; stepsize = 0.1, max_drift_fraction = 0.15, group_steps = 1, adjusted = true)
    stepsize, max_drift_fraction = promote(stepsize, max_drift_fraction)
    return LocalLangevin(stepsize, max_drift_fraction, max(1, Int(group_steps)), adjusted)
end

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

@inline function _mala_log_kernel(x, mean, four_ηT)
    dx = x - mean
    return -(dx * dx) / four_ηT
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

@inline function Processes.init(langevin::LocalLangevin, context::Cont) where {Cont}
    (;model) = context

    hamiltonian = model.hamiltonian
    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    dH_prealloc = zeros(eltype(model), InteractiveIsing.nstates(model))
    active_spins = _active_spin_vector(model)
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

    η = max(stepsize[], eps(SType))
    σ = zero(SType)

    return (;model, hamiltonian, rng, dH_prealloc, active_spins,
                layer_views, stepsize, max_drift_fraction, group_steps,
                adjusted, proposal, ΔE, accepted, attempted, T, η, σ,
                gradient_max, gradient_rms, reflected_fraction)
end

@inline update!(::LocalLangevin, hterm, model::AbstractIsingGraph, proposal::FlipProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function Processes.step!(langevin::LocalLangevin, context::C) where {C}
    (;hamiltonian, rng, model, dH_prealloc, layer_views, stepsize,
        max_drift_fraction, group_steps, adjusted) = context

    SType = eltype(model)
    spins = @inline InteractiveIsing.graphstate(model)
    epsT = eps(SType)
    T = temp(model)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    n_group_steps = max(1, group_steps[])
    use_adjusted = adjusted[]
    dh = d_iH()
    active_spins = _active_spin_vector(model)

    for spin_idx in active_spins
        derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
        derivative = _finite_derivative(SType(derivative))
        @inbounds dH_prealloc[spin_idx] = derivative
    end

    σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
    n = length(active_spins)
    n == 0 && return (;)
    attempted = 0
    accepted = 0
    ΔE = zero(SType)
    four_ηT = SType(4) * η * max(t, epsT)
    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
    gradient_sumsq = zero(SType)
    gradient_count = 0
    reflected = 0

    for _ in 1:n_group_steps
        offset = rand(rng, 0:(n - 1))
        for j in 1:n
            k = j + offset
            if k > n
                k -= n
            end

            spin_idx = @inbounds active_spins[k]
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = _finite_derivative(SType(derivative))
            @inbounds dH_prealloc[spin_idx] = derivative
            gradient_sumsq += derivative * derivative
            gradient_count += 1

            local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, layer_views)
            low_state = local_states[1]
            high_state = local_states[end]
            local_span = high_state - low_state

            local_drift_cap = drift_fraction * local_span
            raw_drift_step = η * derivative
            drift_step = use_adjusted ? raw_drift_step : _langevin_drift_step(η, derivative, local_drift_cap)

            noise = σ > zero(SType) ? randn(rng, SType) : zero(SType)
            old_state = @inbounds spins[spin_idx]
            trial_state = old_state - drift_step + σ * noise
            layer_idx = spin_idx_to_layer_idx(spin_idx, layer_views)
            attempted += 1

            if !use_adjusted
                new_state = _reflect_to_bounds(trial_state, low_state, high_state)
                reflected += new_state == trial_state ? 0 : 1
                proposal = FlipProposal{SType}(spin_idx, old_state, new_state, layer_idx, true)
                @inbounds spins[spin_idx] = new_state
                @inline update!(langevin, hamiltonian, model, proposal)
                post_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
                @inbounds dH_prealloc[spin_idx] = _finite_derivative(SType(post_derivative))
                accepted += 1
                continue
            end

            if !_in_bounds(trial_state, low_state, high_state)
                proposal = FlipProposal{SType}(spin_idx, old_state, old_state, layer_idx, false)
                continue
            end

            new_state = trial_state
            proposal_trial = FlipProposal{SType}(spin_idx, old_state, new_state, layer_idx, false)
            ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal_trial)

            if t <= zero(SType)
                accept_move = isfinite(ΔE) && ΔE <= zero(SType)
            else
                forward_mean = old_state - drift_step

                @inbounds spins[spin_idx] = new_state
                reverse_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
                reverse_derivative = _finite_derivative(SType(reverse_derivative))
                reverse_drift_step = η * reverse_derivative
                reverse_mean = new_state - reverse_drift_step
                @inbounds spins[spin_idx] = old_state

                log_forward_q = _mala_log_kernel(new_state, forward_mean, four_ηT)
                log_reverse_q = _mala_log_kernel(old_state, reverse_mean, four_ηT)
                log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
                accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(SType) || log(rand(rng, SType)) < log_acceptance)
            end

            if accept_move
                proposal = FlipProposal{SType}(spin_idx, old_state, new_state, layer_idx, true)
                @inbounds spins[spin_idx] = new_state
                @inline update!(langevin, hamiltonian, model, proposal)
                post_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
                @inbounds dH_prealloc[spin_idx] = _finite_derivative(SType(post_derivative))
                accepted += 1
            else
                proposal = FlipProposal{SType}(spin_idx, old_state, new_state, layer_idx, false)
            end
        end

    end

    acceptance_rate = attempted == 0 ? zero(SType) : SType(accepted) / SType(attempted)
    gradient_max = isempty(active_spins) ? zero(SType) : maximum(abs, @view dH_prealloc[active_spins])
    gradient_rms = gradient_count == 0 ? zero(SType) : sqrt(gradient_sumsq / SType(gradient_count))
    reflected_fraction = attempted == 0 ? zero(SType) : SType(reflected) / SType(attempted)
    return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
        group_steps = n_group_steps, gradient_max, gradient_rms, reflected_fraction)
end
