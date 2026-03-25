export LangevinDynamics
struct LangevinDynamics <: IsingMCAlgorithm end

@inline function IsingLangevin()
    return Unique(LangevinDynamics())
end

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

    if x < lo
        y = lo + (lo - x)
        if y <= hi
            return y
        end
    else
        y = hi - (x - hi)
        if y >= lo
            return y
        end
    end

    y = mod(x - lo, T(2) * span)
    return y <= span ? lo + y : hi - (y - span)
end

@inline function Processes.init(::LangevinDynamics, context::Cont) where {Cont}
    (;state) = context

    hamiltonian = state.hamiltonian
    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, state)
    dH_prealloc = zeros(eltype(state), InteractiveIsing.nstates(state))
    active_spins = sampling_indices(getfield(state, :index_set))
    layer_views = layers(state)
    T = temp(state)
    SType = eltype(state)
    stepsize = Ref(SType(0.1))
    max_substep = Ref(SType(0.01))
    max_drift_fraction = Ref(SType(0.15))
    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)

    # Init η, η_sub, σ_sub, n_substeps
    η = max(stepsize[], eps(SType))
    ηmax = max(max_substep[], eps(SType))
    σ_sub = zero(SType)
    n_substeps_eta = ceil(Int, η / ηmax)
    n_substeps = max(1, n_substeps_eta)
    η_sub = η / SType(n_substeps)

    return (;state, hamiltonian, rng, dH_prealloc, active_spins, 
                layer_views, stepsize, max_substep, max_drift_fraction, 
                proposal, T, η, ηmax, σ_sub, n_substeps, η_sub)
end

@inline update!(::LangevinDynamics, hterm, state::AbstractIsingGraph, proposal::FlipProposal) = update!(Metropolis(), hterm, state, proposal)

@inline function Processes.step!(langevin::LangevinDynamics, context::C) where {C}
    (;hamiltonian, rng, state, dH_prealloc, active_spins, layer_views, stepsize, max_substep, max_drift_fraction, proposal, T, η, ηmax, σ_sub, n_substeps) = context

    SType = eltype(state)
    spins = @inline InteractiveIsing.graphstate(state)
    epsT = eps(SType)
    T = temp(state)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    ηmax = max(max_substep[], epsT)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))

    max_drift_ratio = zero(SType)
    for spin_idx in active_spins
        derivative = @inbounds dH_prealloc[spin_idx]
        local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, layer_views)
        local_span = local_states[end] - local_states[1]
        local_cap = drift_fraction * local_span
        drift_ratio = abs(η * derivative) / max(local_cap, epsT)
        max_drift_ratio = max(max_drift_ratio, drift_ratio)
    end

    n_substeps_eta = ceil(Int, η / ηmax)
    n_substeps_drift = ceil(Int, max_drift_ratio)
    n_substeps = max(1, n_substeps_eta, n_substeps_drift)
    η_sub = η / SType(n_substeps)
    σ_sub = t > zero(SType) ? sqrt(SType(2) * η_sub * t) : zero(SType)
    n = length(active_spins)
    n == 0 && return (;)
    dh = d_iH()

    for _ in 1:n_substeps
        offset = rand(rng, 0:(n - 1))
        for j in 1:n
            k = j + offset
            if k > n
                k -= n
            end

            spin_idx = @inbounds active_spins[k]
            derivative = @inline calculate(dh, hamiltonian, state, spin_idx)
            if !isfinite(derivative)
                derivative = zero(SType)
            end
            @inbounds dH_prealloc[spin_idx] = derivative

            local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, layer_views)
            low_state = local_states[1]
            high_state = local_states[end]
            local_span = high_state - low_state

            local_drift_cap = drift_fraction * local_span
            drift_step = clamp(η_sub * derivative, -local_drift_cap, local_drift_cap)

            noise = σ_sub > zero(SType) ? randn(rng, SType) : zero(SType)
            old_state = @inbounds spins[spin_idx]
            trial_state = old_state - drift_step + σ_sub * noise


            if !isfinite(trial_state)
                trial_state = old_state
                if !isfinite(trial_state)
                    trial_state = (low_state + high_state) / SType(2)
                end
            end
            new_state = _reflect_to_bounds(trial_state, low_state, high_state)
            layer_idx = spin_idx_to_layer_idx(spin_idx, layer_views)
            proposal = FlipProposal{SType}(spin_idx, old_state, new_state, layer_idx, true)
            @inbounds spins[spin_idx] = new_state
            @inline update!(langevin, hamiltonian, state, proposal)
        end

    end

    return (;proposal, T, η, η_sub, σ_sub, n_substeps)
end
