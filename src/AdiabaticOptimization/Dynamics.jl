# AI Generated

export AdiabaticOptimization, SimulatedBifurcation, SimulatedBifurcationTracked,
    adiabatic_done, bifurcation_done

"""
    AdiabaticOptimization(; backend=:cpu, steps=1000, dt=0.05, pump_start=0,
                          pump_stop=1, damping=0, sync_every=1)

Backend-driven adiabatic optimization process for finding low-energy Ising-like
states.
"""
struct AdiabaticOptimization{B,T<:AbstractFloat} <: ProcessAlgorithm
    backend::B
    steps::Int
    dt::T
    pump_start::T
    pump_stop::T
    damping::T
    sync_every::Int
end

"""
    AdiabaticOptimization(; kwargs...)

Construct an adiabatic optimization process with normalized backend and numeric
schedule parameters.
"""
function AdiabaticOptimization(;
    backend = :cpu,
    steps::Integer = 1000,
    dt::Real = 0.05,
    pump_start::Real = 0,
    pump_stop::Real = 1,
    damping::Real = 0,
    sync_every::Integer = 1,
)
    steps > 0 || throw(ArgumentError("AdiabaticOptimization requires steps > 0."))
    sync_every > 0 || throw(ArgumentError("AdiabaticOptimization requires sync_every > 0."))
    T = typeof(float(promote(dt, pump_start, pump_stop, damping)[1]))
    return AdiabaticOptimization(
        optimization_backend(backend),
        Int(steps),
        T(dt),
        T(pump_start),
        T(pump_stop),
        T(damping),
        Int(sync_every),
    )
end

"""
    SimulatedBifurcation(; kwargs...)

Construct the simulated-bifurcation-compatible preset of `AdiabaticOptimization`.
"""
function SimulatedBifurcation(; kwargs...)
    return AdiabaticOptimization(; kwargs...)
end

"""
    SimulatedBifurcationTracked(; kwargs...)

Compatibility constructor for tracked simulated-bifurcation process names.
"""
function SimulatedBifurcationTracked(; kwargs...)
    return SimulatedBifurcation(; kwargs...)
end

"""
    adiabatic_done(context)

Return the adiabatic completion flag stored in a process context.
"""
function adiabatic_done(context::C) where {C}
    return get(context, :adiabatic_done, false)
end

"""
    bifurcation_done(context)

Return the simulated-bifurcation compatibility completion flag.
"""
function bifurcation_done(context::C) where {C}
    return get(context, :bifurcation_done, adiabatic_done(context))
end

"""
    initial_adiabatic_position!(x, model)

Encode graph state values into the continuous adiabatic position vector.
"""
function initial_adiabatic_position!(x::AbstractVector{T}, model::M) where {T<:AbstractFloat,M<:AbstractIsingGraph}
    spins = graphstate(model)
    for layer in layers(model)
        low = T(stateset(layer)[1])
        high = T(stateset(layer)[end])
        midpoint = (low + high) / T(2)
        for spin_idx in graphidxs(layer)
            @inbounds x[spin_idx] = T(spins[spin_idx]) >= midpoint ? one(T) : -one(T)
        end
    end
    return x
end

"""
    active_adiabatic_mask(model)

Collect a graph-indexed mask for positions that optimization may write back.
"""
function active_adiabatic_mask(model::M) where {M<:AbstractIsingGraph}
    active_mask = falses(nstates(model))
    for spin_idx in sampling_indices(model)
        @inbounds active_mask[Int(spin_idx)] = true
    end
    return active_mask
end

"""
    write_adiabatic_state!(model, x, active_mask)

Write signs of the continuous position vector back to layer endpoint states.
"""
function write_adiabatic_state!(
    model::M,
    x::AbstractVector{T},
    active_mask::AbstractVector{Bool},
) where {M<:AbstractIsingGraph,T<:AbstractFloat}
    spins = graphstate(model)
    for layer in layers(model)
        low = stateset(layer)[1]
        high = stateset(layer)[end]
        for spin_idx in graphidxs(layer)
            @inbounds active_mask[Int(spin_idx)] || continue
            @inbounds spins[spin_idx] = x[spin_idx] < zero(T) ? low : high
        end
    end
    return model
end

"""
    sync_adiabatic_state!(model, x_host, buffers, active_mask)

Copy backend position data to the CPU and synchronize graph state values.
"""
function sync_adiabatic_state!(
    model::M,
    x_host::AbstractVector{T},
    buffers::OptimizationBuffers,
    active_mask::AbstractVector{Bool},
) where {M<:AbstractIsingGraph,T<:AbstractFloat}
    copy_optimization_position!(x_host, buffers)
    write_adiabatic_state!(model, x_host, active_mask)
    return model
end

"""
    adiabatic_oscillator_pump(algo, step)

Evaluate the linear oscillator pump schedule for one optimizer step.
"""
function adiabatic_oscillator_pump(algo::AdiabaticOptimization{B,T}, step::Integer) where {B,T<:AbstractFloat}
    progress = T(step) / T(getfield(algo, :steps))
    return getfield(algo, :pump_start) + progress * (getfield(algo, :pump_stop) - getfield(algo, :pump_start))
end

"""
    run_adiabatic_dynamics!(x, p, grad, pump, dt, damping)

Advance position and momentum buffers by one adiabatic oscillator step.
"""
function run_adiabatic_dynamics!(
    x::AbstractVector{T},
    p::AbstractVector{T},
    grad::AbstractVector{T},
    pump_value::T,
    dt::T,
    damping::T,
) where {T<:AbstractFloat}
    @. p = (one(T) - damping) * p + dt * ((pump_value - one(T)) * x - grad)
    @. x = x + dt * p
    @. p = ifelse(abs(x) > one(T), zero(T), p)
    @. x = clamp(x, -one(T), one(T))
    return x, p
end

"""
    StatefulAlgorithms.init(algo, context)

Initialize Hamiltonian state, GPU buffers, and completion flags for adiabatic
optimization.
"""
function StatefulAlgorithms.init(algo::AdiabaticOptimization, context::Cont) where {Cont}
    (; model) = context
    hamiltonian = init!(model.hamiltonian, model)
    T = eltype(model)
    x0 = initial_adiabatic_position!(Vector{T}(undef, nstates(model)), model)
    buffers = init_optimization_buffers(getfield(algo, :backend), x0)
    x_host = similar(x0)
    active_mask = active_adiabatic_mask(model)
    adiabatic_step = 0
    adiabatic_done = false
    bifurcation_done = false
    pump_value = T(getfield(algo, :pump_start))
    return (; model, hamiltonian, buffers, x_host, active_mask, adiabatic_step,
        adiabatic_done, bifurcation_done, pump = pump_value)
end

"""
    StatefulAlgorithms.step!(algo, context)

Run one adiabatic optimization step and update completion flags.
"""
function StatefulAlgorithms.step!(algo::AdiabaticOptimization, context::C) where {C}
    (; model, hamiltonian, buffers, x_host, active_mask, adiabatic_step, adiabatic_done) = context
    adiabatic_done && return (; adiabatic_step, adiabatic_done, bifurcation_done = true, pump = context.pump)

    next_step = adiabatic_step + 1
    T = eltype(model)
    pump_value = T(adiabatic_oscillator_pump(algo, next_step))
    calculate!(
        getfield(buffers, :grad),
        d_sH(),
        hamiltonian,
        model;
        backend = getfield(buffers, :backend),
        step = next_step,
        total_steps = getfield(algo, :steps),
    )
    run_adiabatic_dynamics!(
        getfield(buffers, :x),
        getfield(buffers, :p),
        getfield(buffers, :grad),
        pump_value,
        T(getfield(algo, :dt)),
        T(getfield(algo, :damping)),
    )

    done = next_step >= getfield(algo, :steps)
    if done || next_step % getfield(algo, :sync_every) == 0
        sync_adiabatic_state!(model, x_host, buffers, active_mask)
    end

    return (; adiabatic_step = next_step, adiabatic_done = done, bifurcation_done = done, pump = pump_value)
end
