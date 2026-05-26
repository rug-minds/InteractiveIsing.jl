export SetGraphState!, CopyGraphState!, RandomizeGraphState!,
       GeometricTemperatureSchedule, ReverseAnnealTemperatureSchedule,
       ResetBestEnergyCapture!, CaptureBestEnergyState!, graph_energy

# Process tool that writes a stored state vector into an Ising graph.
@ProcessAlgorithm function SetGraphState!(isinggraph, target)
    state(isinggraph) .= target
    return nothing
end

# Process tool that copies the current graph state into a reusable buffer.
@ProcessAlgorithm function CopyGraphState!(dest::AbstractVector, isinggraph)
    dest .= state(isinggraph)
    return nothing
end

# Process tool that initializes every active graph state to a random Ising sign.
@ProcessAlgorithm function RandomizeGraphState!(isinggraph, rng::AbstractRNG)
    s = state(isinggraph)
    @inbounds for idx in eachindex(s)
        s[idx] = rand(rng, Bool) ? one(eltype(s)) : -one(eltype(s))
    end
    return nothing
end

"""Return the graph energy for Bilinear + MagField Ising graphs."""
function graph_energy(isinggraph::G) where {G}
    s = state(isinggraph)
    b = getparam(isinggraph.hamiltonian, InteractiveIsing.MagField, :b)
    A = adj(isinggraph)
    colptrs = SparseArrays.getcolptr(A)
    rowvals = SparseArrays.rowvals(A)
    nzvals = SparseArrays.nonzeros(A)
    energy = zero(eltype(s))
    @inbounds for col in 1:size(A, 2)
        for ptr in colptrs[col]:(colptrs[col + 1] - 1)
            energy -= eltype(s)(0.5) * nzvals[ptr] * s[rowvals[ptr]] * s[col]
        end
    end
    @inbounds for idx in eachindex(s)
        energy -= b[idx] * s[idx]
    end
    return energy
end

"""
    GeometricTemperatureSchedule(; start_T, stop_T)

Process scheduler that writes a geometric temperature schedule into a graph.
Reset its `step_idx` owned field before reusing it for a new phase.
"""
@ProcessAlgorithm begin
    @config start_T::Float32 = 5f0
    @config stop_T::Float32 = 1f-2

    function GeometricTemperatureSchedule(
        isinggraph,
        @managed(step_idx = 0),
        @managed(total_steps = n_steps);
        @inputs((; n_steps::Int = 1))
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = start_T * (stop_T / start_T)^progress
        InteractiveIsing.temp!(isinggraph, current_T)
        return (; step_idx = min(step_idx + 1, total - 1), current_T)
    end
end

"""
    ReverseAnnealTemperatureSchedule(; cold_T, peak_T)

Process scheduler that warms from `cold_T` to `peak_T`, then cools back down.
Reset its `step_idx` owned field before reusing it for a new phase.
"""
@ProcessAlgorithm begin
    @config cold_T::Float32 = 1f-2
    @config peak_T::Float32 = 1f0

    function ReverseAnnealTemperatureSchedule(
        isinggraph,
        @managed(step_idx = 0),
        @managed(total_steps = n_steps);
        @inputs((; n_steps::Int = 1))
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = if progress <= 0.5f0
            cold_T + (progress / 0.5f0) * (peak_T - cold_T)
        else
            peak_T + ((progress - 0.5f0) / 0.5f0) * (cold_T - peak_T)
        end
        InteractiveIsing.temp!(isinggraph, current_T)
        return (; step_idx = min(step_idx + 1, total - 1), current_T)
    end
end

# Process tool that prepares a reusable best-energy state buffer.
@ProcessAlgorithm function ResetBestEnergyCapture!(
    best_energy::Base.RefValue,
    best_state::AbstractVector,
)
    best_energy[] = convert(typeof(best_energy[]), Inf)
    fill!(best_state, zero(eltype(best_state)))
    return nothing
end

# Process tool that stores the current graph state when its energy improves.
@ProcessAlgorithm function CaptureBestEnergyState!(
    isinggraph,
    best_energy::Base.RefValue,
    best_state::AbstractVector,
)
    energy = graph_energy(isinggraph)
    if energy < best_energy[]
        best_energy[] = energy
        best_state .= state(isinggraph)
    end
    return (; energy)
end
