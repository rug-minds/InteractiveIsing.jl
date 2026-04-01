@ProcessAlgorithm function CaptureState(isinggraph)
    captured .= state(isinggraph)
    return 
end

Capturer() = Unique(CaptureState())