mutable struct GraphData
    # Magnetic field
    const bfield::Vector{Float32}

    #Beta clamp
    clampparam::Float32
    #clamps
    const clamps::Vector{Float32}
    
end

GraphData(g) = 
    GraphData(
        # bfield
        zeros(Float32, nStates(g)),
        # Clamp Param
        0, 
        # Clamp values
        zeros(Float32, nStates(g))
    )

function reset!(data::GraphData)
    data.bfield .= 0
    data.clampparam = 1
    data.clamps .= 0
end

# mutable struct LayerData
#     # Magnetic field
#     bfield::Base.ReshapedArray

#     #clamps
#     clamps::Base.ReshapedArray
# end

# LayerData(data::GraphData, start, length, width) = 
#     LayerData(
#         reshapeView(data.bfield, start, length, width),
#         reshapeView(data.clamps, start, length, width)    
#         )


function Base.resize!(data::GraphData, newsize, lidxs = nothing)
    oldsize = length(data.bfield)
    sizediff = newsize - oldsize
    if sizediff == 0
        return
    elseif sizediff > 0
        resize!(data.bfield, newsize)
        resize!(data.clamps, newsize)
        data.bfield[oldsize+1:end] .= 0
        data.clamps[oldsize+1:end] .= 0
    else # sizediff < 0
        deleteat!(data.bfield, lidxs)
        deleteat!(data.clamps, lidxs)
    end
    return data
end