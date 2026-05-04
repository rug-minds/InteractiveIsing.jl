export BlockLangevin, DynamicBlockLangevin

"""
    BlockLangevin(; stepsize=0.1, max_drift_fraction=0.15, block_size=256, group_steps=1, adjusted=false)

Block Langevin Monte Carlo update.

Each proposal updates a random cyclic block of active spins and represents the
trial as a `MultiSpinProposal`. This is a compromise between `LocalLangevin`
and `GlobalLangevin`: it avoids moving the whole graph coherently while still
moving more than one spin per proposal.

`adjusted` is a type parameter because it changes the structure of each step:
`adjusted=true` evaluates the MALA correction, while `adjusted=false` uses the
fast reflected always-accepted proposal.

`stepsize` is the proposal size. If a `stepsize` variable is supplied in the
process context, it overrides this configured default at initialization.
"""
struct BlockLangevin{Adjusted,T<:Real} <: IsingMCAlgorithm
    stepsize::T
    max_drift_fraction::T
    block_size::Int
    group_steps::Int

    function BlockLangevin{Adjusted,T}(
        stepsize::T,
        max_drift_fraction::T,
        block_size::Int,
        group_steps::Int,
    ) where {Adjusted,T<:Real}
        Adjusted === true || Adjusted === false ||
            throw(ArgumentError("BlockLangevin adjusted must be true or false, got $(repr(Adjusted))."))
        return new{Adjusted,T}(stepsize, max_drift_fraction, max(1, block_size), max(1, group_steps))
    end
end

function BlockLangevin(; stepsize = 0.1, max_drift_fraction = 0.15, block_size = 256, group_steps = 1, adjusted::Bool = false)
    return BlockLangevin{adjusted}(; stepsize, max_drift_fraction, block_size, group_steps)
end

function BlockLangevin{Adjusted}(; stepsize = 0.1, max_drift_fraction = 0.15, block_size = 256, group_steps = 1) where {Adjusted}
    stepsize, max_drift_fraction = promote(stepsize, max_drift_fraction)
    return BlockLangevin{Adjusted,typeof(stepsize)}(stepsize, max_drift_fraction, Int(block_size), Int(group_steps))
end

BlockLangevin(adjusted::Bool) = BlockLangevin(; adjusted)

"""
    DynamicBlockLangevin(; stepsize=0.1, max_drift_fraction=0.15, max_blocksize=256, group_steps=1, adjusted=false)

Block Langevin update with a fresh random block size for each proposal.

The maximum block size is a type parameter: `DynamicBlockLangevin{MaxBlockSize,
Adjusted,T}`. Each proposal draws a block size uniformly from
`1:min(MaxBlockSize, n_active)` and then chooses a random cyclic block of that
size. Since the size draw is independent of the model state, it cancels from the
Metropolis ratio in the adjusted path.
"""
struct DynamicBlockLangevin{MaxBlockSize,Adjusted,T<:Real} <: IsingMCAlgorithm
    stepsize::T
    max_drift_fraction::T
    group_steps::Int

    function DynamicBlockLangevin{MaxBlockSize,Adjusted,T}(
        stepsize::T,
        max_drift_fraction::T,
        group_steps::Int,
    ) where {MaxBlockSize,Adjusted,T<:Real}
        MaxBlockSize isa Integer && MaxBlockSize >= 1 ||
            throw(ArgumentError("DynamicBlockLangevin max block size must be a positive integer, got $(repr(MaxBlockSize))."))
        Adjusted === true || Adjusted === false ||
            throw(ArgumentError("DynamicBlockLangevin adjusted must be true or false, got $(repr(Adjusted))."))
        return new{MaxBlockSize,Adjusted,T}(stepsize, max_drift_fraction, max(1, group_steps))
    end
end

function DynamicBlockLangevin(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    max_blocksize = 256,
    group_steps = 1,
    adjusted::Bool = false,
)
    max_blocksize = Int(max_blocksize)
    return DynamicBlockLangevin{max_blocksize,adjusted}(; stepsize, max_drift_fraction, group_steps)
end

function DynamicBlockLangevin{MaxBlockSize}(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    group_steps = 1,
    adjusted::Bool = false,
) where {MaxBlockSize}
    return DynamicBlockLangevin{MaxBlockSize,adjusted}(; stepsize, max_drift_fraction, group_steps)
end

function DynamicBlockLangevin{MaxBlockSize,Adjusted}(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    group_steps = 1,
) where {MaxBlockSize,Adjusted}
    stepsize, max_drift_fraction = promote(stepsize, max_drift_fraction)
    return DynamicBlockLangevin{MaxBlockSize,Adjusted,typeof(stepsize)}(stepsize, max_drift_fraction, Int(group_steps))
end

DynamicBlockLangevin(adjusted::Bool) = DynamicBlockLangevin(; adjusted)

@inline function Processes.init(langevin::BlockLangevin{Adjusted}, context::Cont) where {Adjusted,Cont}
    (;model) = context

    for layer in layers(model)
        statetype(layer) isa Discrete &&
            error("BlockLangevin requires Continuous layers; layer $(layeridx(layer)) is Discrete. " *
                  "Use a Metropolis or Heatbath algorithm for discrete spin models.")
    end

    hamiltonian = init!(model.hamiltonian, model)
    rng = Random.MersenneTwister()

    nstates_model = InteractiveIsing.nstates(model)
    active_spins = _active_spin_vector(model)
    layer_views = layers(model)
    SType = eltype(model)

    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(@inline _langevin_unwrap_ref(@inline _langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    block_size = Ref(langevin.block_size)
    group_steps = Ref(langevin.group_steps)
    adjusted = Adjusted

    dH_prealloc = zeros(SType, nstates_model)
    old_vals = Vector{SType}(undef, nstates_model)
    new_vals = Vector{SType}(undef, nstates_model)
    derivatives = Vector{SType}(undef, nstates_model)
    reverse_derivatives = Vector{SType}(undef, nstates_model)
    layer_idxs = Vector{Int}(undef, nstates_model)
    block_idxs = Vector{Int}(undef, min(nstates_model, max(1, block_size[])))

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

@inline function Processes.init(langevin::DynamicBlockLangevin{MaxBlockSize,Adjusted}, context::Cont) where {MaxBlockSize,Adjusted,Cont}
    (;model) = context

    for layer in layers(model)
        statetype(layer) isa Discrete &&
            error("DynamicBlockLangevin requires Continuous layers; layer $(layeridx(layer)) is Discrete. " *
                  "Use a Metropolis or Heatbath algorithm for discrete spin models.")
    end

    hamiltonian = init!(model.hamiltonian, model)
    rng = Random.MersenneTwister()

    nstates_model = InteractiveIsing.nstates(model)
    active_spins = _active_spin_vector(model)
    layer_views = layers(model)
    SType = eltype(model)

    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(@inline _langevin_unwrap_ref(@inline _langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    max_blocksize = MaxBlockSize
    group_steps = Ref(langevin.group_steps)
    adjusted = Adjusted

    dH_prealloc = zeros(SType, nstates_model)
    old_vals = Vector{SType}(undef, nstates_model)
    new_vals = Vector{SType}(undef, nstates_model)
    derivatives = Vector{SType}(undef, nstates_model)
    reverse_derivatives = Vector{SType}(undef, nstates_model)
    layer_idxs = Vector{Int}(undef, nstates_model)
    block_idxs = Vector{Int}(undef, min(nstates_model, MaxBlockSize))

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
        max_drift_fraction, max_blocksize, group_steps, adjusted, dH_prealloc,
        old_vals, new_vals, derivatives, reverse_derivatives, layer_idxs,
        block_idxs, proposal, ΔE, accepted, attempted, acceptance_rate, T, η,
        σ, gradient_max, gradient_rms, reflected_fraction)
end

@inline update!(::BlockLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)
@inline update!(::DynamicBlockLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function _fill_langevin_block!(block_idxs, active_spins, rng, m::Int)
    n = length(active_spins)
    offset = @inline rand(rng, 0:(n - 1))
    @inbounds for pos in 1:m
        k = pos + offset
        while k > n
            k -= n
        end
        block_idxs[pos] = active_spins[k]
    end
    return @view block_idxs[1:m]
end

@inline function _draw_langevin_block_size(::BlockLangevin, context, rng, n_active::Int)
    return min(max(1, context.block_size[]), n_active)
end

@inline function _draw_langevin_block_size(::DynamicBlockLangevin{MaxBlockSize}, context, rng, n_active::Int) where {MaxBlockSize}
    return @inline rand(rng, 1:min(MaxBlockSize, n_active))
end

@inline function Processes.step!(langevin::BlockLangevin{Adjusted}, context::C) where {Adjusted,C}
    return @inline _block_langevin_step!(langevin, context, Val(Adjusted))
end

@inline function Processes.step!(langevin::DynamicBlockLangevin{MaxBlockSize,Adjusted}, context::C) where {MaxBlockSize,Adjusted,C}
    return @inline _block_langevin_step!(langevin, context, Val(Adjusted))
end

@inline function _block_langevin_step!(langevin, context::C, ::Val{Adjusted}) where {Adjusted,C}
    (;hamiltonian, rng, model, layer_views, stepsize,
        max_drift_fraction, group_steps, dH_prealloc,
        old_vals, new_vals, derivatives, reverse_derivatives, layer_idxs,
        block_idxs) = context

    SType = eltype(model)
    spins = @inline InteractiveIsing.graphstate(model)
    epsT = eps(SType)
    T = @inline temp(model)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    n_group_steps = max(1, group_steps[])
    active_spins = @inline _active_spin_vector(model)
    n_active = length(active_spins)
    n_active == 0 && return (;)
    dh = d_iH()

    attempted = 0
    accepted = 0
    reflected = 0
    proposed_spins = 0
    gradient_sumsq = zero(SType)
    gradient_count = 0
    ΔE = zero(SType)
    four_ηT = SType(4) * η * max(t, epsT)
    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)

    for _ in 1:n_group_steps
        m = @inline _draw_langevin_block_size(langevin, context, rng, n_active)
        proposed_spins += m
        if length(block_idxs) < m
            resize!(block_idxs, m)
        end
        idxs = _fill_langevin_block!(block_idxs, active_spins, rng, m)
        in_bounds = true
        log_forward_q = zero(SType)

        @inbounds for (pos, spin_idx) in enumerate(idxs)
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = @inline _finite_derivative(SType(derivative))
            dH_prealloc[spin_idx] = derivative
            derivatives[pos] = derivative
            gradient_sumsq += derivative * derivative
            gradient_count += 1

            local_states = @inline spin_idx_layer_dispatch(stateset, spin_idx, layer_views)
            low_state = local_states[1]
            high_state = local_states[end]
            local_span = high_state - low_state
            local_drift_cap = drift_fraction * local_span
            drift_step = Adjusted ? η * derivative : (@inline _langevin_drift_step(η, derivative, local_drift_cap))

            old_state = spins[spin_idx]
            trial_state = old_state - drift_step + (σ > zero(SType) ? (@inline randn(rng, SType)) * σ : zero(SType))
            new_state = Adjusted ? trial_state : (@inline _reflect_to_bounds(trial_state, low_state, high_state))

            old_vals[pos] = old_state
            new_vals[pos] = new_state
            layer_idxs[pos] = @inline spin_idx_to_layer_idx(spin_idx, layer_views)
            reflected += (!Adjusted && new_state != trial_state) ? 1 : 0

            if Adjusted
                in_bounds &= @inline _in_bounds(trial_state, low_state, high_state)
                log_forward_q += @inline _mala_log_kernel(new_state, old_state - drift_step, four_ηT)
            end
        end

        old_view = @view old_vals[1:m]
        new_view = @view new_vals[1:m]
        layer_view = @view layer_idxs[1:m]
        attempted += 1

        if Adjusted && !in_bounds
            proposal = MultiSpinProposal(idxs, old_view, old_view, layer_view, false)
            continue
        end

        proposal_trial = MultiSpinProposal(idxs, old_view, new_view, layer_view, false)

        if !Adjusted
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
                reverse_derivative = @inline _finite_derivative(SType(reverse_derivative))
                reverse_derivatives[pos] = reverse_derivative
                reverse_mean = new_view[pos] - η * reverse_derivative
                log_reverse_q += @inline _mala_log_kernel(old_view[pos], reverse_mean, four_ηT)
            end

            @inbounds for pos in 1:m
                spins[idxs[pos]] = old_view[pos]
            end

            log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
            accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(SType) || log(@inline rand(rng, SType)) < log_acceptance)
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
    reflected_fraction = proposed_spins == 0 ? zero(SType) : SType(reflected) / SType(proposed_spins)

    return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
        group_steps = n_group_steps, gradient_max, gradient_rms,
        reflected_fraction)
end
