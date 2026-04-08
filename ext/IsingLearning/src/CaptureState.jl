@ProcessAlgorithm function CaptureState(isinggraph, @managed(captured = zeros(length(state(isinggraph)))); @inputs (;isinggraph))
    captured .= state(isinggraph)
    return 
end

Capturer() = Unique(CaptureState())