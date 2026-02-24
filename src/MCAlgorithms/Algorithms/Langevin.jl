struct LangevinDynamics <: ProcessAlgorithm end

function IsingLangevin()
    destr = DestructureInput()
    package(CompositeAlgorithm(LangevinDynamics(), destr, Route(destr => LangevinDynamics(), :isinggraph => :structure)))
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

function Processes.init(::LangevinDynamics, input)
    (;structure) = input

    state = InteractiveIsing.state(structure)
    hamiltonian = structure.hamiltonian
    adj = InteractiveIsing.adj(structure)
    self = structure.self
    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, structure)
    M = sum(state)
    iterator = InteractiveIsing.iterator(structure)

    dH_prealloc = zeros(eltype(state), length(state))

    rand_alloc = rand(rng, eltype(state), length(dH_prealloc))

    iterator = InteractiveIsing.iterator(structure)

    stepsize = Ref(0.1f0)
    max_substep = Ref(eltype(state)(0.01f0))
    max_drift_fraction = Ref(eltype(state)(0.15f0))

    (;isinggraph = structure, hamiltonian, rng, state, adj, self, dH_prealloc, iterator, stepsize, max_substep, max_drift_fraction)
end

@inline function Processes.step!(::LangevinDynamics, context::C) where {C}
    (;hamiltonian, rng, isinggraph, state, adj, self, dH_prealloc, iterator, stepsize, max_substep, max_drift_fraction) = context

    SType = eltype(state)
    epsT = eps(SType)
    t = max(SType(temp(isinggraph)), zero(SType))
    η = max(stepsize[], epsT)
    ηmax = max(max_substep[], epsT)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    # active_spins = InteractiveIsing.iterator(isinggraph)
    active_spins = iterator

    max_drift_ratio = zero(SType)
    for spin_idx in active_spins
        derivative = @inbounds dH_prealloc[spin_idx]
        local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, isinggraph)
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
    ham_args = (;w = adj, s = state, self, hamiltonian...)
    dh = dH()

    for _ in 1:n_substeps
        offset = rand(rng, 0:(n - 1))
        for j in 1:n
            k = j + offset
            if k > n
                k -= n
            end

            spin_idx = @inbounds active_spins[k]
            derivative = @inline calculate(dh, hamiltonian, ham_args, spin_idx)
            if !isfinite(derivative)
                derivative = zero(SType)
            end
            @inbounds dH_prealloc[spin_idx] = derivative

            # local_span = max(@inbounds(span_prealloc[spin_idx]), epsT)
            local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, isinggraph)
            low_state = local_states[1]
            high_state = local_states[end]
            local_span = high_state - low_state

            local_drift_cap = drift_fraction * local_span
            drift_step = clamp(η_sub * derivative, -local_drift_cap, local_drift_cap)

            noise = σ_sub > zero(SType) ? randn(rng, SType) : zero(SType)
            trial_state = @inbounds(state[spin_idx]) - drift_step + σ_sub * noise


            if !isfinite(trial_state)
                trial_state = @inbounds state[spin_idx]
                if !isfinite(trial_state)
                    
                    trial_state = (low_state + high_state) / SType(2)
                    # trial_state = (@inbounds(lo_prealloc[spin_idx]) + @inbounds(hi_prealloc[spin_idx])) / SType(2)
                end
            end

            # @inbounds state[spin_idx] = _reflect_to_bounds(trial_state, lo_prealloc[spin_idx], hi_prealloc[spin_idx])
            @inbounds state[spin_idx] = _reflect_to_bounds(trial_state, low_state, high_state)
        
        end
    end

    return (;)
end
