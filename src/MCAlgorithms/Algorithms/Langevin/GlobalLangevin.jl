export GlobalLangevin

"""
    GlobalLangevin(; stepsize=0.1, max_drift_fraction=0.15, group_steps=1, adjusted=false)

Multi-spin Langevin Monte Carlo update.

Each proposal updates all active spins together and represents the trial as a
`MultiSpinProposal`. The default `adjusted=false` reflects proposals back into
the state bounds and accepts them directly, which is practical for large bounded
systems but not an exact Boltzmann sampler.

`stepsize` is the proposal size used by each global Langevin proposal.
`group_steps` repeats that full global proposal several times inside one
`Processes.step!` call; it does not divide or reinterpret `stepsize`.

With `adjusted=true`, the whole vector move is accepted or rejected with a
Metropolis-adjusted Langevin acceptance probability. This is the Boltzmann
correct path for the discretized proposal, but it can become effectively static
on large bounded graphs because any out-of-bounds coordinate rejects the whole
move. Use a very small `stepsize`, an interior initial state, or
`LocalLangevin(adjusted=true)` when exact adjusted sampling is the priority.
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
    active_spins = _active_spin_vector(model)
    layer_views = layers(model)
    T = temp(model)
    SType = eltype(model)
    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(_langevin_unwrap_ref(_langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    group_steps = Ref(langevin.group_steps)
    adjusted = Ref(langevin.adjusted)

    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)
    ΔE = zero(SType)
    accepted = 0
    attempted = 0
    gradient_max = zero(SType)
    gradient_rms = zero(SType)
    reflected_fraction = zero(SType)
    old_vals = Vector{SType}(undef, nstates_model)
    new_vals = Vector{SType}(undef, nstates_model)
    derivatives = Vector{SType}(undef, nstates_model)
    reverse_derivatives = Vector{SType}(undef, nstates_model)
    layer_idxs = Vector{Int}(undef, nstates_model)

    η = max(stepsize[], eps(SType))
    σ = zero(SType)

    return (;model, hamiltonian, rng, dH_prealloc, active_spins,
                layer_views, stepsize, max_drift_fraction, group_steps,
                adjusted, proposal, ΔE, accepted, attempted, T, η, σ,
                gradient_max, gradient_rms, reflected_fraction,
                old_vals, new_vals, derivatives,
                reverse_derivatives, layer_idxs)
end

@inline update!(::GlobalLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function Processes.step!(langevin::GlobalLangevin, context::C) where {C}
    (;hamiltonian, rng, model, dH_prealloc, layer_views, stepsize,
        max_drift_fraction, group_steps, adjusted, old_vals, new_vals,
        derivatives, reverse_derivatives, layer_idxs) = context

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
    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)
    gradient_sumsq = zero(SType)
    gradient_count = 0
    reflected = 0

    for _ in 1:n_group_steps
        in_bounds = true
        log_forward_q = zero(SType)

        @inbounds for (pos, spin_idx) in enumerate(active_spins)
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = _finite_derivative(SType(derivative))
            dH_prealloc[spin_idx] = derivative
            derivatives[pos] = derivative
            gradient_sumsq += derivative * derivative
            gradient_count += 1

            local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, layer_views)
            low_state = local_states[1]
            high_state = local_states[end]
            local_span = high_state - low_state
            local_drift_cap = drift_fraction * local_span
            drift_step = use_adjusted ? η * derivative : _langevin_drift_step(η, derivative, local_drift_cap)

            old_state = spins[spin_idx]
            trial_state = old_state - drift_step + (σ > zero(SType) ? randn(rng, SType) * σ : zero(SType))

            old_vals[pos] = old_state
            new_state = use_adjusted ? trial_state : _reflect_to_bounds(trial_state, low_state, high_state)
            new_vals[pos] = new_state
            layer_idxs[pos] = spin_idx_to_layer_idx(spin_idx, layer_views)
            reflected += (!use_adjusted && new_state != trial_state) ? 1 : 0

            if use_adjusted
                in_bounds &= _in_bounds(trial_state, low_state, high_state)
                forward_mean = old_state - drift_step
                log_forward_q += _mala_log_kernel(new_vals[pos], forward_mean, four_ηT)
            end
        end

        idxs_view = @view active_spins[1:n]
        old_view = @view old_vals[1:n]
        new_view = @view new_vals[1:n]
        layer_view = @view layer_idxs[1:n]
        attempted += 1

        if use_adjusted && !in_bounds
            proposal = MultiSpinProposal(idxs_view, old_view, old_view, layer_view, false)
            continue
        end

        proposal_trial = MultiSpinProposal(idxs_view, old_view, new_view, layer_view, false)

        if !use_adjusted
            proposal = MultiSpinProposal(idxs_view, old_view, new_view, layer_view, true)
            @inbounds for pos in 1:n
                spins[idxs_view[pos]] = new_view[pos]
            end
            @inline update!(langevin, hamiltonian, model, proposal)
            accepted += 1
            continue
        end

        ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal_trial)
        if t <= zero(SType)
            accept_move = isfinite(ΔE) && ΔE <= zero(SType)
        else
            @inbounds for pos in 1:n
                spins[idxs_view[pos]] = new_view[pos]
            end

            log_reverse_q = zero(SType)
            @inbounds for pos in 1:n
                spin_idx = idxs_view[pos]
                reverse_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
                reverse_derivative = _finite_derivative(SType(reverse_derivative))
                reverse_derivatives[pos] = reverse_derivative
                reverse_mean = new_view[pos] - η * reverse_derivative
                log_reverse_q += _mala_log_kernel(old_view[pos], reverse_mean, four_ηT)
            end

            @inbounds for pos in 1:n
                spins[idxs_view[pos]] = old_view[pos]
            end

            log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
            accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(SType) || log(rand(rng, SType)) < log_acceptance)
        end

        if accept_move
            proposal = MultiSpinProposal(idxs_view, old_view, new_view, layer_view, true)
            @inbounds for pos in 1:n
                spins[idxs_view[pos]] = new_view[pos]
            end
            @inline update!(langevin, hamiltonian, model, proposal)
            accepted += 1
        else
            proposal = MultiSpinProposal(idxs_view, old_view, new_view, layer_view, false)
        end
    end

    acceptance_rate = attempted == 0 ? zero(SType) : SType(accepted) / SType(attempted)
    gradient_max = isempty(active_spins) ? zero(SType) : maximum(abs, @view dH_prealloc[active_spins])
    gradient_rms = gradient_count == 0 ? zero(SType) : sqrt(gradient_sumsq / SType(gradient_count))
    reflection_denominator = n_group_steps * n
    reflected_fraction = reflection_denominator == 0 ? zero(SType) : SType(reflected) / SType(reflection_denominator)
    return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
        group_steps = n_group_steps, gradient_max, gradient_rms, reflected_fraction)
end
