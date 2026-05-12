@ProcessAlgorithm function ComputeGradients(forward_state, backward_state, @managed(st), @managed(ps), @managed(layer), @input (;st, ps, layer))
    # Buffers are the shape of (;w, b, α) and will be filled with the appropriate gradients.
    
    
end

function LearningStep(layer)
    (;algorithm, plus_capture, minus_capture) = NudgedComp(layer)
    backwards_plus_gradient = 

end