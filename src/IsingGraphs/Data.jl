mutable struct GraphData
    # For tracking defects
    aliveList::Vector{Vert}
    defects::Bool
    const defectBools::Vector{Bool}
    defectList::Vector{Vert}
    
    # Magnetic field
    const mlist::Vector{Float32}

    #Beta clamp
    clampparam::Float32
    #clamps
    const clamps::Vector{Float32}
    
end

GraphData(g) = 
    GraphData(
        # AliveList
        Int32.([1:Nstates(g);]),
        #Defects?
        false, 
        # DefectBools
        [false for x in 1:Nstates(g)], 
        # Defect List
        Vector{Int32}(), 
        # Mlist
        zeros(Float32, Nstates(g)),
        # Clamp Param
        0, 
        # Clamp values
        zeros(Float32, Nstates(g))
    )

function reinitGraphData!(data, g)
    data.aliveList = Int32.([1:Nstates(g);])
    data.defects = false
    data.defectBools .= false
    data.defectList = Vector{Int32}()
    data.mlist .= 0
    data.clamps .= 0
end

struct LayerData
    # Magnetic field
    mlist::Base.ReshapedArray

    #clamps
    clamps::Base.ReshapedArray
end

LayerData(data::GraphData, start, width, length) = 
    LayerData(
        reshapeView(data.mlist, start, width, length),
        reshapeView(data.clamps, start, width, length)
    )
