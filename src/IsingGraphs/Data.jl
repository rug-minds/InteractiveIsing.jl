mutable struct GraphData
    # For tracking defects
    aliveList::Vector{Vert}
    defects::Bool
    defectBools::Vector{Bool}
    defectList::Vector{Vert}
    
    # Magnetic field
    mlist::Vector{Float32}

    #Beta clamp
    clampparam::Float32
    #clamps
    clamps::Vector{Float32}
    
end

GraphData(g) = 
    GraphData(
        # AliveList
        Int32.([1:nStates(g);]),
        #Defects?
        false, 
        # DefectBools
        [false for x in 1:nStates(g)], 
        # Defect List
        Vector{Int32}(), 
        # Mlist
        zeros(Float32, nStates(g)),
        # Clamp Param
        0, 
        # Clamp values
        zeros(Float32, nStates(g))
    )

function reinitGraphData!(data, g)
    data.aliveList = Int32.([1:nStates(g);])
    data.defects = false
    data.defectBools .= false
    data.defectList = Vector{Int32}()
    data.mlist .= 0
    data.clamps .= 0
end

mutable struct LayerData
    # Magnetic field
    mlist::Base.ReshapedArray

    #clamps
    clamps::Base.ReshapedArray

    #number of defects
    ndefects::Int32
end

LayerData(data::GraphData, start, length, width) = 
    LayerData(
        reshapeView(data.mlist, start, length, width),
        reshapeView(data.clamps, start, length, width),
        0
    )
