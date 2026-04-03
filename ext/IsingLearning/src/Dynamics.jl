export ForwardDynamics, NudgedProcess

ForwardDynamics(windowsize, tol,) = CompositeAlgorithm(
    InteractiveIsing.Metropolis(),
    ConvergeanceTest()
)

@ProcessAlgorithm function resetgraph!(isinggraph::G) where G
    resetstate!(isinggraph)
    return 
end

# @ProcessAlgorithm function empty()
#     return
# end

function NudgedProcess(layer)
    beta = layer.β
    fullsweeps = layer.fullsweeps
    n_units = layer.nunits
    # c1 = ConvergeanceTest(windowsize, tol)
    # c2 = ConvergeanceTest(windowsize, tol)
    plus_capture = Capturer()
    minus_capture = Capturer()
    
    plus = @Routine begin
        @alias dynamics = Metropolis()
        @alias plus_capture = plus_capture
        
        state = dynamics()
        @repeat fullsweeps*n_units plus_capture(isinggraph = state)
        resetgraph!(state)
    end
        

    # minus = Routine(CompositeAlgorithm(:dynamics => Metropolis()), :minus_capture => minus_capture, resetgraph!, (Repeat(fullsweeps*n_units), 1,1),
    #     Route(Metropolis => minus_capture, :state => :isinggraph),
    #     Route(Metropolis => resetgraph!, :state => :isinggraph)
    # )

    minus = @Routine begin
        @alias dynamics = Metropolis()
        @alias minus_capture = minus_capture
        
        state = dynamics()
        @repeat fullsweeps*n_units minus_capture(isinggraph = state)
        resetgraph!(state)
    end


    final = @CompositeAlgorithm begin
        @state buffers
        @alias plus = plus
        @alias minus = minus
        @context c1 = plus()
        @context c2 = minus()
        contrastive_gradient(plus_capture = c1.plus_capture.buffer, minus_capture = c2.minus_capture.buffer, β = beta, buffers = buffers) 
    end 
    (;algorithm = final, plus_capture, minus_capture, dynamics = plus.dynamics)
end

