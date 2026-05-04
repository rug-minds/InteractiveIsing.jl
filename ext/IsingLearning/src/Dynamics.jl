export PowerLawTemperatureSchedule, ForwardDynamics, NudgedDynamics, Forward_and_Nudged

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
        @managed(total_steps = n_steps);
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

function apply_input(isinggraph, x)
    InteractiveIsing.off!(isinggraph.index_set, 1)
    state(isinggraph[1]) .= x
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
    fill!(clamping.y, zero(eltype(clamping.y)))
    clamping.y[output_idxs] .= y
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
    relaxation_steps = layer.relaxation_steps
    n_units = layer.nunits
  
    forward = @Routine begin
        @alias dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        initstate!(dynamics.model)
        apply_input(dynamics.model, x)
        model = @repeat relaxation_steps dynamics()
        copyvector!(equilibrium_state, @transform(x -> InteractiveIsing.state(x), model))
    end

    (;algorithm = forward, dynamics = forward.dynamics)
end

function NudgedDynamics(layer)
    beta = layer.β
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    relaxation_steps = layer.relaxation_steps
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
    nudged = NudgedDynamics(layer).algorithm
    beta = layer.β

    final = @CompositeAlgorithm begin
        @state buffers

        @context c1 = forward()
        @context c2 = nudged()

        # Reset clamping after backward phases
        set_clamping_beta!(c1.dynamics.model, zero(beta))

        contrastive_gradient(c1.dynamics.model, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers) 
    end 
    (;algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end
