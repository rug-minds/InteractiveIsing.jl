export ForwardDynamics, NudgedProcess

ForwardDynamics(windowsize, tol,) = CompositeAlgorithm(
    InteractiveIsing.Metropolis(),
    ConvergeanceTest()
)

@ProcessAlgorithm function setgraph!(isinggraph::G, target) where G
    # resetstate!(isinggraph)
    state(isinggraph) .= target
    return 
end

# @ProcessAlgorithm function empty()
#     return
# end

function NudgedProcess(layer)
    beta = layer.β
    fullsweeps = layer.fullsweeps
    n_units = layer.nunits
  
    plus_capture = Capturer()
    minus_capture = Capturer()
    
    plus = @Routine begin
        @state equilibrium_state
        @alias dynamics = Metropolis()
        @alias plus_capture = plus_capture
        
        setgraph!(isinggraph = state, target = equilibrium_state)
        state = @repeat fullsweeps*n_units dynamics()
        plus_capture(isinggraph = state)
    end

    minus = @Routine begin
        @state equilibrium_state
        @alias dynamics = Metropolis()
        @alias minus_capture = minus_capture
        
        setgraph!(isinggraph = state, target = equilibrium_state)
        state = @repeat fullsweeps*n_units dynamics()
        minus_capture(isinggraph = state)
    end


    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = plus()
        @context c2 = minus()
        contrastive_gradient(c1.dynamics.state, c1.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers) 
    end 
    (;algorithm = resolve(final), plus_capture, minus_capture, dynamics = plus.dynamics)
end

