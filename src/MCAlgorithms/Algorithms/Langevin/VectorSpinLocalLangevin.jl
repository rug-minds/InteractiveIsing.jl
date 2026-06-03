"""
    _vector_local_langevin_storage(model)

Allocate derivative storage for vector-spin LocalLangevin updates.
"""
@inline function _vector_local_langevin_storage(model::G) where {G<:AbstractVectorSpinGraph}
    return fill(zero(spin_state_type(model)), InteractiveIsing.nstates(model))
end

"""
    _vector_langevin_finite(derivative)

Replace non-finite vector derivative components by zero.
"""
@inline function _vector_langevin_finite(derivative::SVector{D,T}) where {D,T<:AbstractFloat}
    return SVector{D,T}(ntuple(i -> isfinite(derivative[i]) ? derivative[i] : zero(T), Val(D)))
end

"""
    _vector_langevin_metric(derivative)

Return the scalar gradient magnitude used for diagnostics and drift summaries.
"""
@inline _vector_langevin_metric(derivative::SVector) = norm(derivative)

"""
    _vector_langevin_in_bounds(spin, low, high)

Return true when every vector component is inside the layer state bounds.
"""
@inline function _vector_langevin_in_bounds(spin::SVector{D,T}, low::T, high::T) where {D,T<:AbstractFloat}
    return all(i -> @inline(_in_bounds(spin[i], low, high)), 1:D)
end

"""
    _vector_langevin_reflect(spin, low, high)

Reflect every vector component into the layer state bounds.
"""
@inline function _vector_langevin_reflect(spin::SVector{D,T}, low::T, high::T) where {D,T<:AbstractFloat}
    reflected = false
    new_spin = SVector{D,T}(ntuple(Val(D)) do i
        component = @inline _reflect_to_bounds(spin[i], low, high)
        reflected |= component != spin[i]
        component
    end)
    return new_spin, reflected ? 1 : 0
end

"""
    _vector_langevin_drift_step(drift, cap)

Limit every vector drift component to the local layer span fraction.
"""
@inline function _vector_langevin_drift_step(drift::SVector{D,T}, cap::T) where {D,T<:AbstractFloat}
    return SVector{D,T}(ntuple(i -> clamp(drift[i], -cap, cap), Val(D)))
end

"""
    _vector_langevin_boundary_drift_step(drift, old_state, low, high)

Suppress drift components that point out through an active layer boundary.
"""
@inline function _vector_langevin_boundary_drift_step(
    drift::SVector{D,T},
    old_state::SVector{D,T},
    low::T,
    high::T,
) where {D,T<:AbstractFloat}
    return SVector{D,T}(ntuple(Val(D)) do i
        if old_state[i] <= low && drift[i] > zero(T)
            zero(T)
        elseif old_state[i] >= high && drift[i] < zero(T)
            zero(T)
        else
            drift[i]
        end
    end)
end

"""
    _vector_langevin_log_kernel(x, mean, four_ηT)

Return the vector Gaussian log proposal kernel up to an additive constant.
"""
@inline function _vector_langevin_log_kernel(x::SVector, mean::SVector, four_ηT)
    dx = x - mean
    return -dot(dx, dx) / four_ηT
end

"""
    _vector_local_langevin_derivative!(storage, dh, hamiltonian, model, spin_idx)

Evaluate and store one vector-spin local energy derivative.
"""
@inline function _vector_local_langevin_derivative!(
    dH_prealloc,
    dh,
    hamiltonian,
    model::G,
    spin_idx::Int,
) where {G<:AbstractVectorSpinGraph}
    derivative = @inline _vector_langevin_finite(@inline calculate(dh, hamiltonian, model, spin_idx))
    @inbounds dH_prealloc[spin_idx] = derivative
    return derivative
end

"""
    _start_vector_local_langevin_cycle!(...)

Refresh cached vector derivatives at the start of a LocalLangevin sweep.
"""
@inline function _start_vector_local_langevin_cycle!(
    langevin::LocalLangevin{Order},
    context,
    dh,
    hamiltonian,
    model::G,
    active_spins,
) where {Order,G<:AbstractVectorSpinGraph}
    SType = eltype(model)
    gradient_max = zero(SType)
    for spin_idx in active_spins
        derivative = @inline _vector_local_langevin_derivative!(
            context.dH_prealloc,
            dh,
            hamiltonian,
            model,
            spin_idx,
        )
        gradient_max = max(gradient_max, SType(@inline _vector_langevin_metric(derivative)))
    end

    n = length(active_spins)
    context.group_remaining[] = max(1, context.group_steps_ref[])
    context.sweep_position[] = 1
    if Order === :random
        @inline _reset_local_langevin_order!(context.sweep_order, n)
    end
    return gradient_max
end

"""
    _vector_local_langevin_accept!(...)

Commit one accepted vector-spin LocalLangevin proposal and refresh its cached
local derivative.
"""
@inline function _vector_local_langevin_accept!(
    langevin::LocalLangevin,
    dh,
    hamiltonian,
    model::G,
    dH_prealloc,
    spin_idx::Int,
    layer_idx::Int,
    old_state::S,
    new_state::S,
) where {G<:AbstractVectorSpinGraph,S<:SVector}
    spins = @inline InteractiveIsing.graphstate(model)
    proposal = FlipProposal{S}(spin_idx, old_state, new_state, layer_idx, true)
    @inbounds spins[spin_idx] = new_state
    @inline update!(langevin, hamiltonian, model, proposal)
    post_derivative = @inline _vector_local_langevin_derivative!(dH_prealloc, dh, hamiltonian, model, spin_idx)
    return proposal, post_derivative
end

"""
    _vector_local_langevin_reverse_mean(...)

Evaluate the reverse vector Langevin proposal mean for the MALA correction.
"""
@inline function _vector_local_langevin_reverse_mean(
    dh,
    hamiltonian,
    model::G,
    spin_idx::Int,
    old_state::S,
    new_state::S,
    η,
) where {G<:AbstractVectorSpinGraph,S<:SVector}
    spins = @inline InteractiveIsing.graphstate(model)
    @inbounds spins[spin_idx] = new_state
    reverse_derivative = @inline _vector_langevin_finite(@inline calculate(dh, hamiltonian, model, spin_idx))
    reverse_mean = new_state - η * reverse_derivative
    @inbounds spins[spin_idx] = old_state
    return reverse_mean
end

"""
    _vector_local_langevin_adjusted!(...)

Run one Metropolis-adjusted vector-spin LocalLangevin proposal.
"""
@inline function _vector_local_langevin_adjusted!(
    langevin::LocalLangevin,
    rng,
    dh,
    hamiltonian,
    model::G,
    dH_prealloc,
    spin_idx::Int,
    layer_idx::Int,
    old_state::S,
    trial_state::S,
    low_state,
    high_state,
    derivative::S,
    drift_step::S,
    t,
    η,
    four_ηT,
) where {G<:AbstractVectorSpinGraph,S<:SVector}
    if !(@inline _vector_langevin_in_bounds(trial_state, low_state, high_state))
        return FlipProposal{S}(spin_idx, old_state, old_state, layer_idx, false), zero(eltype(model)), 0, derivative
    end

    proposal_trial = FlipProposal{S}(spin_idx, old_state, trial_state, layer_idx, false)
    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal_trial)

    if t <= zero(t)
        accept_move = isfinite(ΔE) && ΔE <= zero(ΔE)
    else
        forward_mean = old_state - drift_step
        reverse_mean = @inline _vector_local_langevin_reverse_mean(
            dh,
            hamiltonian,
            model,
            spin_idx,
            old_state,
            trial_state,
            η,
        )
        log_forward_q = @inline _vector_langevin_log_kernel(trial_state, forward_mean, four_ηT)
        log_reverse_q = @inline _vector_langevin_log_kernel(old_state, reverse_mean, four_ηT)
        log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
        accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(log_acceptance) || log(@inline rand(rng, typeof(t))) < log_acceptance)
    end

    if accept_move
        proposal, post_derivative = @inline _vector_local_langevin_accept!(
            langevin,
            dh,
            hamiltonian,
            model,
            dH_prealloc,
            spin_idx,
            layer_idx,
            old_state,
            trial_state,
        )
        return proposal, ΔE, 1, post_derivative
    end

    return FlipProposal{S}(spin_idx, old_state, trial_state, layer_idx, false), ΔE, 0, derivative
end

"""
    _vector_spin_local_langevin_init(langevin, context)

Initialize LocalLangevin state for vector-spin graphs.
"""
@inline function _vector_spin_local_langevin_init(
    langevin::LocalLangevin{Order,Adjusted},
    context,
) where {Order,Adjusted}
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

    hamiltonian = init!(model.hamiltonian, model)
    rng = Random.MersenneTwister()
    dH_prealloc = @inline _vector_local_langevin_storage(model)
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
        layer_views, stepsize, max_drift_fraction, group_steps_ref, adjusted,
        sweep_position, sweep_order, sweep_offset, group_remaining, T)
end

"""
    Processes.init(langevin::LocalLangevin, context::SubContextView)

Dispatch LocalLangevin initialization to vector-spin storage when the compiled
subcontext model type is an `AbstractVectorSpinGraph`.
"""
@inline @generated function Processes.init(
    langevin::LocalLangevin{Order,Adjusted},
    context::Processes.SubContextView{CType,SubName},
) where {Order,Adjusted,CType,SubName}
    if CType <: Processes.ProcessContext
        subcontexts_type = CType.parameters[1]
        if SubName in fieldnames(subcontexts_type)
            subcontext_type = fieldtype(subcontexts_type, SubName)
            if subcontext_type <: Processes.SubContext
                data_type = subcontext_type.parameters[2]
                if :model in fieldnames(data_type)
                    model_type = fieldtype(data_type, :model)
                    if model_type <: AbstractVectorSpinGraph
                        return :(@inline _vector_spin_local_langevin_init(langevin, context))
                    end
                end
            end
        end
    end
    return :(@inline _scalar_local_langevin_init(langevin, context))
end

"""
    _vector_spin_local_langevin_step!(langevin, context)

Run one vector-spin LocalLangevin update.
"""
@inline function _vector_spin_local_langevin_step!(
    langevin::LocalLangevin{Order,Adjusted},
    context,
) where {Order,Adjusted}
    (;hamiltonian, rng, model, dH_prealloc, layer_views, stepsize,
        max_drift_fraction, T) = context

    SType = eltype(model)
    SpinType = eltype(InteractiveIsing.graphstate(model))
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
        gradient_max = @inline _start_vector_local_langevin_cycle!(
            langevin,
            context,
            dh,
            hamiltonian,
            model,
            active_spins,
        )
    end

    ΔE = zero(SType)
    four_ηT = SType(4) * η * max(t, epsT)
    proposal = FlipProposal{SpinType}(1, zero(SpinType), zero(SpinType), 1, false)
    reflected = 0

    k = @inline _next_local_langevin_order_index!(langevin, context, rng, n)
    spin_idx = @inbounds active_spins[k]
    derivative = @inline _vector_local_langevin_derivative!(dH_prealloc, dh, hamiltonian, model, spin_idx)
    gradient_max = max(gradient_max, SType(@inline _vector_langevin_metric(derivative)))
    low_state, high_state, local_span, layer_idx = @inline _local_langevin_bounds(spin_idx, layer_views)
    low_state = SType(low_state)
    high_state = SType(high_state)
    local_span = SType(local_span)

    local_drift_cap = drift_fraction * local_span
    raw_drift_step = η * derivative
    drift_step = Adjusted ? raw_drift_step : (@inline _vector_langevin_drift_step(raw_drift_step, local_drift_cap))
    old_state = @inbounds spins[spin_idx]
    if !Adjusted
        drift_step = @inline _vector_langevin_boundary_drift_step(drift_step, old_state, low_state, high_state)
    end

    noise_step = if σ > zero(SType)
        σ * SpinType(ntuple(_ -> randn(rng, SType), Val(spin_dimension(model))))
    else
        zero(SpinType)
    end
    trial_state = old_state - drift_step + noise_step
    post_derivative = derivative
    accepted = 0

    if !Adjusted
        new_state, reflected = @inline _vector_langevin_reflect(trial_state, low_state, high_state)
        proposal, post_derivative = @inline _vector_local_langevin_accept!(
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
        accepted += 1
    else
        proposal, ΔE, accepted, post_derivative = @inline _vector_local_langevin_adjusted!(
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
    gradient_max = max(gradient_max, SType(@inline _vector_langevin_metric(post_derivative)))

    @inline _advance_local_langevin_cursor!(langevin, context, n)

    acceptance_rate = SType(accepted)
    gradient_rms = SType(@inline _vector_langevin_metric(derivative))
    reflected_fraction = SType(reflected)

    return (;proposal, ΔE, accepted, acceptance_rate, η, σ,
        gradient_max, gradient_rms, reflected_fraction)
end

"""
    Processes.step!(langevin::LocalLangevin, context::SubContextView)

Dispatch LocalLangevin stepping to the vector-spin kernel when the compiled
subcontext model type is an `AbstractVectorSpinGraph`.
"""
@inline @generated function Processes.step!(
    langevin::LocalLangevin{Order,Adjusted},
    context::Processes.SubContextView{CType,SubName},
) where {Order,Adjusted,CType,SubName}
    if CType <: Processes.ProcessContext
        subcontexts_type = CType.parameters[1]
        if SubName in fieldnames(subcontexts_type)
            subcontext_type = fieldtype(subcontexts_type, SubName)
            if subcontext_type <: Processes.SubContext
                data_type = subcontext_type.parameters[2]
                if :model in fieldnames(data_type)
                    model_type = fieldtype(data_type, :model)
                    if model_type <: AbstractVectorSpinGraph
                        return :(@inline _vector_spin_local_langevin_step!(langevin, context))
                    end
                end
            end
        end
    end
    return :(@inline _scalar_local_langevin_step!(langevin, context))
end
