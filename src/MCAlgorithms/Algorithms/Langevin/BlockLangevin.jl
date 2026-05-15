export BlockLangevin, DynamicBlockLangevin

"""
    BlockLangevin(; stepsize=0.1, max_drift_fraction=0.15, block_size=256, group_steps=1, adjusted=false)

Block-gradient Langevin Monte Carlo update.

At the start of each internal cycle, one `Processes.step!` refreshes the
derivative on a random group of active spins. That step and subsequent
steps then consume the cached block derivatives one spin at a time. This is a
compromise between `LocalLangevin` and `GlobalLangevin`: it avoids a full
gradient refresh while keeping the per-step state mutation to one spin.

`adjusted` is a type parameter because it changes the structure of each cycle:
`adjusted=true` accepts or rejects the whole block proposal with a MALA
correction, while `adjusted=false` uses the fast always-accepted single-spin
stream with deterministic drift clamped to bounds and stochastic displacements
reflected.

`stepsize` is the proposal size. If a `stepsize` variable is supplied in the
process context, it overrides this configured default at initialization.
`group_steps` is retained for interface compatibility; a `Processes.step!` call
still attempts only one spin update.
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

Block Langevin update with a fresh random block size for each step.

The maximum block size is a type parameter: `DynamicBlockLangevin{MaxBlockSize,
Adjusted,T}`. Each proposal draws a block size uniformly from
`1:min(MaxBlockSize, n_active)` and then chooses a random group of that size.
In adjusted mode the selected block is accepted or rejected as one block
proposal, then accepted entries are written one spin per `Processes.step!`.
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
    rng = @inline _langevin_context_value(context, :rng, Random.MersenneTwister())

    nstates_model = InteractiveIsing.nstates(model)
    active_index_set = index_set(model)
    active_spins = collect(@inline _active_spin_vector(active_index_set))
    layer_views = layers(model)
    SType = eltype(model)

    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(@inline _langevin_unwrap_ref(@inline _langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    block_size = Ref(langevin.block_size)
    group_steps = Ref(langevin.group_steps)
    adjusted = Adjusted

    dH_prealloc = zeros(SType, nstates_model)
    derivatives = Vector{SType}(undef, nstates_model)
    old_vals = Vector{SType}(undef, nstates_model)
    new_vals = Vector{SType}(undef, nstates_model)
    layer_idxs = Vector{Int}(undef, nstates_model)
    block_idxs = Vector{Int}(undef, min(nstates_model, max(1, block_size[])))
    block_shuffle_position = Ref(length(active_spins) + 1)

    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
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
    schedule_position = Ref(0)
    schedule_length = Ref(0)
    schedule_accepted = Ref(false)
    schedule_ΔE = Ref(zero(SType))
    gradient_max_cache = Ref(zero(SType))
    gradient_sumsq_cache = Ref(zero(SType))

    return (;model, hamiltonian, rng, active_index_set, active_spins, layer_views, stepsize,
        max_drift_fraction, block_size, group_steps, adjusted, dH_prealloc,
        derivatives, old_vals, new_vals, layer_idxs, block_idxs, block_shuffle_position,
        proposal, ΔE, accepted, attempted, acceptance_rate, T, η,
        σ, gradient_max, gradient_rms, reflected_fraction, schedule_position,
        schedule_length, schedule_accepted, schedule_ΔE, gradient_max_cache, gradient_sumsq_cache)
end

@inline function Processes.init(langevin::DynamicBlockLangevin{MaxBlockSize,Adjusted}, context::Cont) where {MaxBlockSize,Adjusted,Cont}
    (;model) = context

    for layer in layers(model)
        statetype(layer) isa Discrete &&
            error("DynamicBlockLangevin requires Continuous layers; layer $(layeridx(layer)) is Discrete. " *
                  "Use a Metropolis or Heatbath algorithm for discrete spin models.")
    end

    hamiltonian = init!(model.hamiltonian, model)
    rng = @inline _langevin_context_value(context, :rng, Random.MersenneTwister())

    nstates_model = InteractiveIsing.nstates(model)
    active_index_set = index_set(model)
    active_spins = collect(@inline _active_spin_vector(active_index_set))
    layer_views = layers(model)
    SType = eltype(model)

    stepsize_default = SType(langevin.stepsize)
    stepsize = Ref(SType(@inline _langevin_unwrap_ref(@inline _langevin_context_value(context, :stepsize, stepsize_default))))
    max_drift_fraction = Ref(SType(langevin.max_drift_fraction))
    max_blocksize = MaxBlockSize
    group_steps = Ref(langevin.group_steps)
    adjusted = Adjusted

    dH_prealloc = zeros(SType, nstates_model)
    derivatives = Vector{SType}(undef, nstates_model)
    old_vals = Vector{SType}(undef, nstates_model)
    new_vals = Vector{SType}(undef, nstates_model)
    layer_idxs = Vector{Int}(undef, nstates_model)
    block_idxs = Vector{Int}(undef, min(nstates_model, MaxBlockSize))
    block_shuffle_position = Ref(length(active_spins) + 1)

    proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
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
    schedule_position = Ref(0)
    schedule_length = Ref(0)
    schedule_accepted = Ref(false)
    schedule_ΔE = Ref(zero(SType))
    gradient_max_cache = Ref(zero(SType))
    gradient_sumsq_cache = Ref(zero(SType))

    return (;model, hamiltonian, rng, active_index_set, active_spins, layer_views, stepsize,
        max_drift_fraction, max_blocksize, group_steps, adjusted, dH_prealloc,
        derivatives, old_vals, new_vals, layer_idxs, block_idxs, block_shuffle_position,
        proposal, ΔE, accepted, attempted, acceptance_rate, T, η,
        σ, gradient_max, gradient_rms, reflected_fraction, schedule_position,
        schedule_length, schedule_accepted, schedule_ΔE, gradient_max_cache, gradient_sumsq_cache)
end

@inline update!(::BlockLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)
@inline update!(::DynamicBlockLangevin, hterm, model::AbstractIsingGraph, proposal::AbstractProposal) = update!(Metropolis(), hterm, model, proposal)

@inline function _fill_langevin_block!(
    block_idxs::Vector{Int},
    active_spins::Vector{Int},
    block_shuffle_position::Base.RefValue{Int},
    rng::Random.AbstractRNG,
    m::Int,
)
    n = length(active_spins)
    first_pos = block_shuffle_position[]
    if first_pos < 1 || first_pos + m - 1 > n
        @inbounds for pos in 1:(n - 1)
            swap_pos = @inline rand(rng, pos:n)
            active_spins[pos], active_spins[swap_pos] = active_spins[swap_pos], active_spins[pos]
        end
        first_pos = 1
    end
    @inbounds for pos in 1:m
        block_idxs[pos] = active_spins[first_pos + pos - 1]
    end
    block_shuffle_position[] = first_pos + m
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
        derivatives, old_vals, new_vals, layer_idxs, block_idxs, block_shuffle_position,
        schedule_position, schedule_length, schedule_accepted, schedule_ΔE,
        gradient_max_cache, gradient_sumsq_cache) = context

    SType = eltype(model)
    epsT = eps(SType)
    T = @inline temp(model)
    t = max(SType(T), zero(SType))
    η = max(stepsize[], epsT)
    σ = t > zero(SType) ? sqrt(SType(2) * η * t) : zero(SType)
    drift_fraction = clamp(max_drift_fraction[], epsT, one(SType))
    n_group_steps = max(1, group_steps[])
    active_changed = @inline consume_changed!(context.active_index_set)
    if active_changed
        @inline _set_local_langevin_active_spins!(context.active_spins, @inline _active_spin_vector(context.active_index_set))
        block_shuffle_position[] = length(context.active_spins) + 1
    end
    active_spins = context.active_spins
    n_active = length(active_spins)
    if n_active == 0
        schedule_position[] = 0
        schedule_length[] = 0
        proposal = FlipProposal{SType}(1, zero(SType), zero(SType), 1, false)
        return (;proposal, ΔE = zero(SType), accepted = 0, attempted = 0,
            acceptance_rate = zero(SType), T, η, σ, group_steps = n_group_steps,
            block_size = 0, refreshed_gradient = false,
            gradient_max = zero(SType), gradient_rms = zero(SType),
            reflected_fraction = zero(SType))
    end
    dh = d_iH()

    if Adjusted
        if schedule_accepted[] && schedule_position[] > 0 && schedule_position[] <= schedule_length[]
            pos = schedule_position[]
            spin_idx = @inbounds block_idxs[pos]
            proposal = @inline _langevin_accept_single_spin!(
                langevin,
                hamiltonian,
                model,
                spin_idx,
                @inbounds(layer_idxs[pos]),
                @inbounds(old_vals[pos]),
                @inbounds(new_vals[pos]),
            )
            schedule_position[] = pos + 1
            accepted = 1
            attempted = 1
            acceptance_rate = one(SType)
            gradient_max = gradient_max_cache[]
            gradient_rms = schedule_length[] == 0 ? zero(SType) : sqrt(gradient_sumsq_cache[] / SType(schedule_length[]))
            return (;proposal, ΔE = schedule_ΔE[], accepted, attempted, acceptance_rate, T, η, σ,
                group_steps = n_group_steps, block_size = schedule_length[],
                refreshed_gradient = false, gradient_max, gradient_rms,
                reflected_fraction = zero(SType))
        end

        m = @inline _draw_langevin_block_size(langevin, context, rng, n_active)
        if length(block_idxs) < m
            resize!(block_idxs, m)
        end
        @inline _fill_langevin_block!(block_idxs, active_spins, block_shuffle_position, rng, m)

        gradient_max = zero(SType)
        gradient_sumsq = zero(SType)
        log_forward_q = zero(SType)
        in_bounds = true
        four_ηT = SType(4) * η * max(t, epsT)
        spins = @inline InteractiveIsing.graphstate(model)
        @inbounds for pos in 1:m
            spin_idx = block_idxs[pos]
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = @inline _finite_derivative(SType(derivative))
            dH_prealloc[spin_idx] = derivative
            derivatives[pos] = derivative
            gradient_sumsq += derivative * derivative
            gradient_max = max(gradient_max, abs(derivative))
            low_state, high_state, _, layer_idx = @inline _local_langevin_bounds(spin_idx, layer_views)
            old_state = spins[spin_idx]
            drift_step = η * derivative
            new_state = old_state - drift_step + (σ > zero(SType) ? (@inline randn(rng, SType)) * σ : zero(SType))
            old_vals[pos] = old_state
            new_vals[pos] = new_state
            layer_idxs[pos] = layer_idx
            in_bounds &= @inline _in_bounds(new_state, low_state, high_state)
            log_forward_q += @inline _mala_log_kernel(new_state, old_state - drift_step, four_ηT)
        end

        ΔE = zero(SType)
        accept_move = false
        if in_bounds
            @inbounds for pos in 1:m
                fp = FlipProposal{SType}(block_idxs[pos], old_vals[pos], new_vals[pos], layer_idxs[pos], false)
                ΔE += @inline calculate(ΔH(), hamiltonian, model, fp)
                spins[block_idxs[pos]] = new_vals[pos]
            end
            @inbounds for pos in 1:m
                spins[block_idxs[pos]] = old_vals[pos]
            end
            if t <= zero(SType)
                accept_move = isfinite(ΔE) && ΔE <= zero(SType)
            else
                @inbounds for pos in 1:m
                    spins[block_idxs[pos]] = new_vals[pos]
                end
                log_reverse_q = zero(SType)
                @inbounds for pos in 1:m
                    spin_idx = block_idxs[pos]
                    reverse_derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
                    reverse_derivative = @inline _finite_derivative(SType(reverse_derivative))
                    reverse_mean = new_vals[pos] - η * reverse_derivative
                    log_reverse_q += @inline _mala_log_kernel(old_vals[pos], reverse_mean, four_ηT)
                end
                @inbounds for pos in 1:m
                    spins[block_idxs[pos]] = old_vals[pos]
                end
                log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
                accept_move = isfinite(log_acceptance) && (log_acceptance >= zero(SType) || log(@inline rand(rng, SType)) < log_acceptance)
            end
        end

        schedule_position[] = 1
        schedule_length[] = accept_move ? m : 0
        schedule_accepted[] = accept_move
        schedule_ΔE[] = ΔE
        gradient_max_cache[] = gradient_max
        gradient_sumsq_cache[] = gradient_sumsq

        if accept_move
            spin_idx = @inbounds block_idxs[1]
            proposal = @inline _langevin_accept_single_spin!(
                langevin,
                hamiltonian,
                model,
                spin_idx,
                @inbounds(layer_idxs[1]),
                @inbounds(old_vals[1]),
                @inbounds(new_vals[1]),
            )
            schedule_position[] = 2
            accepted = 1
        else
            proposal = FlipProposal{SType}(@inbounds(block_idxs[1]), @inbounds(old_vals[1]), @inbounds(new_vals[1]), @inbounds(layer_idxs[1]), false)
            accepted = 0
        end

        attempted = 1
        acceptance_rate = SType(accepted)
        gradient_rms = sqrt(gradient_sumsq / SType(m))
        return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
            group_steps = n_group_steps, block_size = m, refreshed_gradient = true,
            gradient_max, gradient_rms, reflected_fraction = zero(SType))
    end

    refreshed = active_changed || schedule_position[] == 0 || schedule_position[] > schedule_length[]
    if refreshed
        m = @inline _draw_langevin_block_size(langevin, context, rng, n_active)
        if length(block_idxs) < m
            resize!(block_idxs, m)
        end
        @inline _fill_langevin_block!(block_idxs, active_spins, block_shuffle_position, rng, m)

        gradient_max = zero(SType)
        gradient_sumsq = zero(SType)
        @inbounds for pos in 1:m
            spin_idx = block_idxs[pos]
            derivative = @inline calculate(dh, hamiltonian, model, spin_idx)
            derivative = @inline _finite_derivative(SType(derivative))
            dH_prealloc[spin_idx] = derivative
            derivatives[pos] = derivative
            gradient_sumsq += derivative * derivative
            gradient_max = max(gradient_max, abs(derivative))
        end
        schedule_position[] = 1
        schedule_length[] = m
        gradient_max_cache[] = gradient_max
        gradient_sumsq_cache[] = gradient_sumsq
    end

    pos = schedule_position[]
    spin_idx = @inbounds block_idxs[pos]
    derivative = @inbounds derivatives[pos]
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
        Val(Adjusted),
    )
    schedule_position[] = pos + 1

    attempted = 1
    acceptance_rate = attempted == 0 ? zero(SType) : SType(accepted) / SType(attempted)
    gradient_max = gradient_max_cache[]
    gradient_rms = schedule_length[] == 0 ? zero(SType) : sqrt(gradient_sumsq_cache[] / SType(schedule_length[]))
    reflected_fraction = SType(reflected)

    return (;proposal, ΔE, accepted, attempted, acceptance_rate, T, η, σ,
        group_steps = n_group_steps, block_size = schedule_length[],
        refreshed_gradient = refreshed, gradient_max, gradient_rms, reflected_fraction)
end
