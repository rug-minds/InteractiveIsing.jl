mutable struct GraphData
    # Magnetic field
    mlist::Vector{Float32}

    #Beta clamp
    clampparam::Float32
    #clamps
    clamps::Vector{Float32}
    
end

GraphData(g) = 
    GraphData(
        # Mlist
        zeros(Float32, nStates(g)),
        # Clamp Param
        0, 
        # Clamp values
        zeros(Float32, nStates(g))
    )

function reset!(data::GraphData)
    data.mlist .= 0
    data.clampparam = 1
    data.clamps .= 0
end

mutable struct LayerData
    # Magnetic field
    mlist::Base.ReshapedArray

    #clamps
    clamps::Base.ReshapedArray
end

LayerData(data::GraphData, start, length, width) = 
    LayerData(
        reshapeView(data.mlist, start, length, width),
        reshapeView(data.clamps, start, length, width)    
        )


function resize!(data::GraphData, newsize)
    resize!(data.mlist, newsize)
    resize!(data.clamps, newsize)
end