
#TODO: Should probably make this a dict
Base.@kwdef mutable struct Obs{T}
    # Active Layer
    layerIdx::Observable{Int32} = Observable(Int32(1))
    
    # currentLayer::

    # Name of the selected layer
    layerName::Observable{String} = Observable("")
    #number of layers
    nlayers::Observable{Int32} = Observable(Int32(0))
    # length/width of graph
    qmllength::Observable{Int32} = Observable(Int32(0))
    
    qmlwidth::Observable{Int32} = Observable(Int32(0))

    # Temperature Observable
    temp::Observable{T} = Observable(1f0)

    # For drawing to simulation
    brush::Observable{T} = Observable(0f0)
    brushR::Observable{Int32} = Observable(Int32(0))

    # Magnetization
    M::Observable{T} = Observable(0f0)

    # Updates per frame average
    upf::Observable{T} = Observable(0f0)
    # Updates per frame per unit
    upfps::Observable{T} = Observable(0f0)

    # Size of Image Window QML
    imgSize::Observable{Tuple{Int64, Int64}} = Observable((0,0))

    # Analysis Is running
    analysisRunning::Observable{Bool} = Observable(false)

    # Is Paused
    isPaused::Observable{Bool} = Observable(false)
    
    # Run the timed functions
    runTimedFunctions::Observable{Bool} = Observable(true)

    data::Dict{Symbol, Observable} = Dict{Symbol, Observable}()

end

Obs(len, wid, brushR, initTemp) = Obs(
    qmllength = Observable(Int32(len)),
    qmlwidth = Observable(Int32(wid)),
    brushR = Observable(Int32(brushR)),
    temp = Observable(initTemp) 
)
