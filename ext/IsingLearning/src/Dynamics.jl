export ForwardDynamics, NudgedDynamics, Forwards_and_Nudged

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

function apply_input(state, x)
    # Set the iterator (don't vary inputs) and apply to inputindexes...
end

function apply_targets(state, y)
    # Set the clamping hamiltonian for the target indexes...
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
        apply_input(dynamics.state, y)
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
        apply_input(dynamics.state, y)
        apply_targets(dynamics.state, y)
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
        apply_input(dynamics.state, y)
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

function Forwards_and_Nudged(layer)
    forward = ForwardDynamics(layer).algorithm
    nudged = NudgedDynamics(layer).algorithm

    final = @CompositeAlgorithm begin
        @state buffers

        @context c1 = forward()
        @context c2 = nudged()

        contrastive_gradient(c1.dynamics.state, c2.plus_capture.captured, c2.minus_capture.captured, layer.β, buffers = buffers) 
    end 
    (;algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

