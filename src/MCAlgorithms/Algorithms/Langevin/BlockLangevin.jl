export BlockLangevin

"""
    BlockLangevin(; stepsize=0.1, max_drift_fraction=0.15, block_size=256, group_steps=1, adjusted=false)

Block Langevin Monte Carlo update.

Each proposal updates a random cyclic block of active spins and represents the
trial as a `MultiSpinProposal`. This is a compromise between `LocalLangevin`
and `GlobalLangevin`: it avoids moving the whole graph coherently while still
moving more than one spin per proposal.

`stepsize` is the proposal size. If a `stepsize` variable is supplied in the
process context, it overrides this configured default at initialization.
"""
struct BlockLangevin{T<:Real} <: IsingMCAlgorithm
    stepsize::T
    max_drift_fraction::T
    block_size::Int
    group_steps::Int
    adjusted::Bool
end

function BlockLangevin(; stepsize = 0.1, max_drift_fraction = 0.15, block_size = 256, group_steps = 1, adjusted = false)
    stepsize, max_drift_fraction = promote(stepsize, max_drift_fraction)
    return BlockLangevin(stepsize, max_drift_fraction, max(1, Int(block_size)), max(1, Int(group_steps)), adjusted)
end

BlockLangevin(adjusted::Bool) = BlockLangevin(; adjusted)

@inline function Processes.init(langevin::BlockLangevin, context::Cont) where {Cont}
    (;model) = context

    hamiltonian = init!(model.hamiltonian, model)
    rng = Random.MersenneTwister()

    nstates_model = InteractiveIsing.nstates(model)
    active_spins = _active_spin_vector(model)
    layer_views = layers(model)
    SType = eltype(model)

    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(_langevin_unwrap_ref(_langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    block_size = Ref(langevin.block_size)
    group_steps = Ref(langevin.group_steps)
    adjusted = Ref(langevin.adjusted)

    dH_prealloc = zeros(SType, nstates_model)
    old_vals = Vector{SType}(undef, nstates_model)
    new_vals = Vector{SType}(undef, nstates_model)
    derivatives = Vector{SType}(undef, nstates_model)
    reverse_derivatives = Vector{SType}(undef, nstates_model)
    layer_idxs = Vector{Int}(undef, nstates_model)
    block_idxs = Vector{Int}(undef, min(length(active_spins), max(1, block_size[])))

    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)
    ΔE = zero(SType)
    accepted = 0
    attempted = 0
    T = temp(model)
    η = max(stepsize[], eps(SType))
    σ = zero(SType)
    acceptance_rate = zero(SType)
    gradient_max = zero(SType)
    gradient_rms = zero(SType)
    reflected_fraction = zero(SType)

    return (;model, hamiltonian, rng, active_spins, layer_views, stepsize,
        max_drift_fraction, block_size, group_steps, adjusted, dH_prealloc,
        old_vals, new_vals, derivatives, reverse_derivatives, layer_idxs,
        block_idxs, proposal, ΔE, accepted, attempted, acceptance_rate, T, η,
        σ, gradient_max, gradient_rms, reflected_fraction)
end

@inline update!(::BlockLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function _fill_langevin_block!(block_idxs, active_spins, rng, m::Int)
    n = length(active_spins)
    offset = rand(rng, 0:(n - 1))
    @inbounds for pos in 1:m
        k = pos + offset
        while k > n
            k -= n
        end
        block_idxs[pos] = active_spins[k]
    end
    return @view block_idxs[1:m]
end

@inline function Processes.step!(langevin::BlockLangevin, context::C) where {C}
    (;hamiltonian, rng, model, active_spins, layer_views, stepsize,
        max_drift_fraction, block_size, group_steps, adjusted, dH_prealloc,
        old_vals, new_vals, derivatives, reverse_derivatives, layer_idxs,
        block_idxs) = context

    SType = eltype(model)
    spins = @inline InteractiveIsing.graphstate(model)
    epsT = eps(SType)
    T = temp(model)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    n_group_steps = max(1, group_steps[])
    use_adjusted = adjusted[]
    n_active = length(active_spins)
    n_active == 0 && return (;)
    m = min(max(1, block_size[]), n_active)
    dh = d_iH()

    attempted = 0
    accepted = 0
    reflected = 0
    gradient_sumsq = zero(SType)
    gradient_count = 0
    ΔE = zero(SType)
    four_ηT = SType(4) * η * max(t, epsT)
    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)

    for _ in 1:n_group_steps
        idxs = _fill_langevin_block!(block_idxs, active_spins, rng, m)
        in_bounds = true
        log_forward_q = zero(SType)

        @inbounds for (pos, spin_idx) in enumerate(idxs)
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
            new_state = use_adjusted ? trial_state : _reflect_to_bounds(trial_state, low_state, high_state)

            old_vals[pos] = old_state
            new_vals[pos] = new_state
            layer_idxs[pos] = spin_idx_to_layer_idx(spin_idx, layer_views)
            reflected += (!use_adjusted && new_state != trial_state) ? 1 : 0

            if use_adjusted
                in_bounds &= _in_bounds(trial_state, low_state, high_state)
                log_forward_q += _mala_log_kernel(new_state, old_state - drift_step, four_ηT)
            end
        end

        old_view = @view old_vals[1:m]
        new_view = @view new_vals[1:m]
        layer_view = @view layer_idxs[1:m]
        attempted += 1

        if use_adjusted && !in_bounds
            proposal = MultiSpinProposal(idxs, old_view, old_view, layer_view, false)
            continue
        end

        proposal_trial = MultiSpinProposal(idxs, old_view, new_view, layer_view, false)

        if !use_adjusted
            proposal = MultiSpinProposal(idxs, old_view, new_view, layer_view, true)
            @inbounds for pos in 1:m
                spins[idxs[pos]] = new_view[pos]
            end
            @inline update!(langevin, hamiltonian, model, proposal)
            accepted += 1
            continue
        end

        ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal_trial)
        if t <= zero(SType)
            accept_move = isfinite(ΔE) && ΔE <= zero(SType)
        else
            @inbounds for pos in 1:m
                spins[idxs[pos]] = new_view[pos]
            end

            log_reverse_q = zero(SType)
            @inbounds for pos in 1:m
                spin_idx = idxs[pos]
                reverse_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
                reverse_derivative = _finite_derivative(SType(reverse_derivative))
                reverse_derivatives[pos] = reverse_derivative
                reverse_mean = new_view[pos] - η * reverse_derivative
                log_reverse_q += _mala_log_kernel(old_view[pos], reverse_mean, four_ηT)
            end

            @inbounds for pos in 1:m
                spins[idxs[pos]] = old_view[pos]
            end

            log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
            accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(SType) || log(rand(rng, SType)) < log_acceptance)
        end

        if accept_move
            proposal = MultiSpinProposal(idxs, old_view, new_view, layer_view, true)
            @inbounds for pos in 1:m
                spins[idxs[pos]] = new_view[pos]
            end
            @inline update!(langevin, hamiltonian, model, proposal)
            accepted += 1
        else
            proposal = MultiSpinProposal(idxs, old_view, new_view, layer_view, false)
        end
    end

    acceptance_rate = attempted == 0 ? zero(SType) : SType(accepted) / SType(attempted)
    gradient_max = isempty(active_spins) ? zero(SType) : maximum(abs, @view dH_prealloc[active_spins])
    gradient_rms = gradient_count == 0 ? zero(SType) : sqrt(gradient_sumsq / SType(gradient_count))
    reflected_fraction = (n_group_steps * m) == 0 ? zero(SType) : SType(reflected) / SType(n_group_steps * m)

    return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
        group_steps = n_group_steps, gradient_max, gradient_rms,
        reflected_fraction)
end
