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
    output_layer = isinggraph[end]
    output_idxs = InteractiveIsing.layerrange(output_layer)
    clamping = isinggraph.hamiltonian[InteractiveIsing.Clamping]
    fill!(clamping.y, zero(eltype(clamping.y)))
    clamping.y[output_idxs] .= y
    return isinggraph
end

function set_clamping_beta!(isinggraph, β)
    clamping = isinggraph.hamiltonian[InteractiveIsing.Clamping]
    clamping.β[] = β
    return isinggraph
end


function ForwardDynamics(layer)
    beta = layer.β
    fullsweeps = layer.fullsweeps
    n_units = layer.nunits
  
    forward = @Routine begin
        @alias dynamics = Metropolis()
        @state equilibrium_state = zeros(n_units)
        @state x

        initstate!(dynamics.state)
        apply_input(dynamics.state, x)
        state = @repeat fullsweeps*n_units dynamics()
        copyvector!(equilibrium_state, @transform(x -> InteractiveIsing.state(x), state))
    end

    (;algorithm = forward, dynamics = forward.dynamics)
end

function NudgedDynamics(layer)
    beta = layer.β
    fullsweeps = layer.fullsweeps
    n_units = layer.nunits
  
    plus_capture = Capturer()
    minus_capture = Capturer()
    
    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = Metropolis()
        @alias plus_capture = plus_capture
        
        setgraph!(isinggraph = dynamics.state, target = equilibrium_state)
        apply_input(dynamics.state, x)
        apply_targets(dynamics.state, y)
        set_clamping_beta!(dynamics.state, beta)
        state = @repeat fullsweeps*n_units dynamics()
        plus_capture(isinggraph = state)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = Metropolis()
        @alias minus_capture = minus_capture
        
        setgraph!(isinggraph = dynamics.state, target = equilibrium_state)
        apply_input(dynamics.state, x)
        apply_targets(dynamics.state, y)
        set_clamping_beta!(dynamics.state, -beta)
        state = @repeat fullsweeps*n_units dynamics()
        minus_capture(isinggraph = state)
    end


    final = @CompositeAlgorithm begin
        @state buffers

        @context c1 = plus()
        @context c2 = minus()

        # contrastive_gradient(c1.dynamics.state, c1.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers) 
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
        set_clamping_beta!(c1.dynamics.state, zero(beta))

        contrastive_gradient(c1.dynamics.state, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers) 
    end 
    (;algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end
