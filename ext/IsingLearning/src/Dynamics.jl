export PowerLawTemperatureSchedule, PowerLawStepsizeSchedule, ForwardDynamics, NudgedDynamics, Forward_and_Nudged

@ProcessAlgorithm function setgraph!(isinggraph::G, target) where G
    # resetstate!(isinggraph)
    state(isinggraph) .= target
    return 
end

function copyvector!(dest::D, src) where D
    dest .= src
    return 
end

@ProcessAlgorithm function initstate!(isinggraph::G) where G
    resetstate!(isinggraph)
    return 
end

function _power_law_temperature(progress::Real, start_T::Real, stop_T::Real, power::Real)
    progress_f32 = clamp(Float32(progress), 0f0, 1f0)
    start_f32 = Float32(start_T)
    stop_f32 = Float32(stop_T)
    power_f32 = Float32(power)

    power_f32 > 0f0 || throw(ArgumentError("power must be positive, got $(power)"))

    # progress = 0 gives start_T, progress = 1 gives stop_T.
    return stop_f32 + (start_f32 - stop_f32) * (1f0 - progress_f32)^power_f32
end

function _power_law_value(progress::Real, start_value::Real, stop_value::Real, power::Real)
    progress_f32 = clamp(Float32(progress), 0f0, 1f0)
    start_f32 = Float32(start_value)
    stop_f32 = Float32(stop_value)
    power_f32 = Float32(power)

    power_f32 > 0f0 || throw(ArgumentError("power must be positive, got $(power)"))
    return stop_f32 + (start_f32 - stop_f32) * (1f0 - progress_f32)^power_f32
end

"""
    PowerLawStepsizeSchedule(; start_stepsize = 1f-2, stop_stepsize = 1f-3, power = 2f0)

Process-algorithm scheduler that returns a routed power-law Langevin
`stepsize` before the sampler is stepped.

This is deliberately separate from the sampler. Use it in a composite next to
the Langevin algorithm and route its `stepsize` output into the sampler context.
"""
@ProcessAlgorithm begin
    @config start_stepsize::Float32 = 1f-2
    @config stop_stepsize::Float32 = 1f-3
    @config power::Float32 = 2f0

    function PowerLawStepsizeSchedule(
        stepsize,
        @managed(step_idx = 0),
        @managed(total_steps = n_steps);
        @inputs((; n_steps::Int = 1))
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        η = _power_law_value(progress, start_stepsize, stop_stepsize, power)
        next_step = min(step_idx + 1, total - 1)
        return (; step_idx = next_step, stepsize = η)
    end
end

"""
    PowerLawTemperatureSchedule(; start_T = 1f0, stop_T = 1f-4, power = 1f0)

Process algorithm that updates `temp(isinggraph)` according to a bounded power-law decay:

`T(p) = stop_T + (start_T - stop_T) * (1 - p)^power`, where `p ∈ [0, 1]`.

Use `n_steps` as an init input to control over how many calls the schedule is traversed.
The first call uses `start_T` and the final call uses `stop_T`.
"""
@ProcessAlgorithm begin
    @config start_T::Float32 = 1f0
    @config stop_T::Float32 = 1f-4
    @config power::Float32 = 1f0

    function PowerLawTemperatureSchedule(
        isinggraph,
        @managed(step_idx = 0),
        @managed(total_steps = n_steps),
        @managed(current_T = start_T);
        @inputs((; n_steps::Int = 1))
    )
        total = max(total_steps, 1)
        progress = total == 1 ? 1f0 : Float32(step_idx) / Float32(total - 1)
        current_T = _power_law_temperature(progress, start_T, stop_T, power)
        InteractiveIsing.temp!(isinggraph, current_T)

        next_step = min(step_idx + 1, total - 1)
        return (; step_idx = next_step, current_T)
    end
end

"""
    _mnist_magfield(isinggraph, field_idx)

Return the `field_idx`th `MagField` in graph Hamiltonian order. MNIST graphs
may use the first field for learnable base biases and the second field for a
worker-local precomputed input pattern.
"""
function _mnist_magfield(isinggraph::G, field_idx::I) where {G,I<:Integer}
    seen = 0
    for hterm in InteractiveIsing.hamiltonians(isinggraph.hamiltonian)
        if hterm isa InteractiveIsing.MagField
            seen += 1
            seen == Int(field_idx) && return hterm
        end
    end
    return nothing
end

"""Return the learnable base MNIST bias field."""
function _mnist_base_magfield(isinggraph::G) where {G}
    field = _mnist_magfield(isinggraph, 1)
    isnothing(field) && error("MNIST graph has no MagField for learnable base biases")
    return field
end

"""Return the optional worker-local MNIST input-pattern field."""
function _mnist_input_magfield(isinggraph::G) where {G}
    return _mnist_magfield(isinggraph, 2)
end

"""
    precompute_mnist_input_pattern!(isinggraph, dest, x)

Write the local-field pattern induced by MNIST input `x` into `dest`. The
result can be reused across free/plus/minus phases for one sample.
"""
function precompute_mnist_input_pattern!(isinggraph::G, dest::D, x) where {G,D<:AbstractVector}
    input_idxs = InteractiveIsing.layerrange(isinggraph[1])
    length(x) == length(input_idxs) ||
        throw(DimensionMismatch("input length $(length(x)) does not match input layer length $(length(input_idxs))"))

    fill!(dest, zero(eltype(dest)))

    adjacency = adj(isinggraph)
    colptrs = SparseArrays.getcolptr(adjacency)
    rowvals = SparseArrays.rowvals(adjacency)
    nzvals = SparseArrays.nonzeros(adjacency)

    @inbounds for (x_idx, graph_idx) in enumerate(input_idxs)
        xval = x[x_idx]
        for ptr in colptrs[graph_idx]:(colptrs[graph_idx + 1] - 1)
            dest[rowvals[ptr]] += nzvals[ptr] * xval
        end
    end
    return dest
end

"""
    apply_input_pattern!(isinggraph, pattern)

Install a precomputed MNIST input field into the graph's second `MagField`.
The input layer is disabled and its state is cleared so the image is not counted
both as fixed spins and as a local field.
"""
function apply_input_pattern!(isinggraph::G, pattern::P) where {G,P<:AbstractVector}
    input_field = _mnist_input_magfield(isinggraph)
    isnothing(input_field) && error("MNIST graph has no second MagField for input patterns")
    length(pattern) == length(input_field.b) ||
        throw(DimensionMismatch("pattern length $(length(pattern)) does not match field length $(length(input_field.b))"))

    InteractiveIsing.off!(isinggraph.index_set, 1)
    input_field.b .= pattern
    input_state = state(isinggraph[1])
    fill!(input_state, zero(eltype(input_state)))
    return isinggraph
end

"""
    apply_input(isinggraph, x)

Install one MNIST input sample. Field-mode graphs write the image-induced local
field directly into the worker-local input `MagField`; state-mode graphs retain
the legacy fixed-input-spin path.
"""
function apply_input(isinggraph, x)
    InteractiveIsing.off!(isinggraph.index_set, 1)
    input_field = _mnist_input_magfield(isinggraph)
    if isnothing(input_field)
        input_state = state(isinggraph[1])
        input_state .= reshape(x, size(input_state))
    else
        precompute_mnist_input_pattern!(isinggraph, input_field.b, x)
    end
    hook = get(isinggraph, :after_apply_input!, nothing)
    isnothing(hook) || hook(isinggraph)
    return isinggraph
end

function apply_targets(isinggraph, y)
    _apply_targets!(_learning_clamping(isinggraph), isinggraph, y)
    return isinggraph
end

function set_clamping_beta!(isinggraph, β)
    clamping = _learning_clamping(isinggraph)
    clamping.β[] = β
    return isinggraph
end

function _find_hamiltonian_term(hts, ::Type{T}) where {T}
    for hterm in InteractiveIsing.hamiltonians(hts)
        hterm isa T && return hterm
    end
    return nothing
end

function _learning_clamping(isinggraph)
    constant_readout = _find_hamiltonian_term(isinggraph.hamiltonian, ConstantLinearReadoutNudge)
    !isnothing(constant_readout) && return constant_readout

    readout_clamping = _find_hamiltonian_term(isinggraph.hamiltonian, LinearReadoutClamping)
    !isnothing(readout_clamping) && return readout_clamping

    direct_clamping = _find_hamiltonian_term(isinggraph.hamiltonian, InteractiveIsing.Clamping)
    !isnothing(direct_clamping) && return direct_clamping

    error("isinggraph has neither LinearReadoutClamping nor InteractiveIsing.Clamping")
end

function _apply_targets!(clamping::InteractiveIsing.Clamping, isinggraph, y)
    output_layer = isinggraph[end]
    output_idxs = InteractiveIsing.layerrange(output_layer)
    length(y) == length(output_idxs) || throw(DimensionMismatch("target length $(length(y)) does not match output layer length $(length(output_idxs))"))
    fill!(clamping.y, zero(eltype(clamping.y)))
    fill!(clamping.mask, zero(eltype(clamping.mask)))
    clamping.y[output_idxs] .= y
    clamping.mask[output_idxs] .= one(eltype(clamping.mask))
    return clamping
end

function _apply_targets!(clamping::LinearReadoutClamping, isinggraph, y)
    isempty(y) && throw(ArgumentError("LinearReadoutClamping needs a scalar target in y[1]"))
    clamping.target[] = first(y)
    return clamping
end

function _apply_targets!(clamping::ConstantLinearReadoutNudge, isinggraph, y)
    isempty(y) && throw(ArgumentError("ConstantLinearReadoutNudge needs a scalar target in y[1]"))
    clamping.target[] = first(y)
    clamping.free_score[] = readout_score(clamping, InteractiveIsing.graphstate(isinggraph))
    return clamping
end


function ForwardDynamics(layer; dynamics_algorithm = layer.dynamics_algorithm)
    beta = layer.β
    dynamics_algorithm = deepcopy(dynamics_algorithm)
    relaxation_steps = layer.free_relaxation_steps
    n_units = layer.nunits
  
    forward = @Routine begin
        @alias dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        resetstate!(dynamics.model)
        apply_input(dynamics.model, x)
        model = @repeat relaxation_steps dynamics()
        copyvector!(equilibrium_state, @transform(x -> InteractiveIsing.state(x), model))
    end

    (;algorithm = forward, dynamics = forward.dynamics)
end

function NudgedDynamics(layer)
    beta = layer.β
    dynamics_algorithm = deepcopy(layer.nudged_dynamics_algorithm)
    relaxation_steps = layer.nudged_relaxation_steps
    n_units = layer.nunits
  
    plus_capture = Capturer()
    minus_capture = Capturer()
    
    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = dynamics_algorithm
        @alias plus_capture = plus_capture
        
        setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        apply_input(dynamics.model, x)
        apply_targets(dynamics.model, y)
        set_clamping_beta!(dynamics.model, beta)
        model = @repeat relaxation_steps dynamics()
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = dynamics_algorithm
        @alias minus_capture = minus_capture
        
        setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        apply_input(dynamics.model, x)
        apply_targets(dynamics.model, y)
        set_clamping_beta!(dynamics.model, -beta)
        model = @repeat relaxation_steps dynamics()
        minus_capture(isinggraph = model)
    end


    final = @CompositeAlgorithm begin
        @state buffers

        @context c1 = plus()
        @context c2 = minus()

        # contrastive_gradient(c1.dynamics.model, c1.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers) 
    end 
    (;algorithm = final, plus_capture, minus_capture, dynamics = plus.dynamics)
end

function Forward_and_Nudged(layer)
    forward = ForwardDynamics(layer).algorithm
    nudged = NudgedDynamics(layer)
    beta = layer.β

    final = @CompositeAlgorithm begin
        @state buffers

        @context c1 = forward()
        @context c2 = nudged.algorithm()

        # Reset clamping after backward phases
        set_clamping_beta!(c1.dynamics.model, zero(beta))

        contrastive_gradient(c1.dynamics.model, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers) 
    end 
    (;algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end
