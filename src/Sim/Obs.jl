struct Obs
    # Active Layer
    activeLayer::Observable{Int32}
    #
    nlayers::Observable{Int32}
    # length/width of graph
    gSize::Observable{Int32}

    # Temperature Observable
    Temp::Observable{Float32}

    # For drawing to simulation
    brush::Observable{Float32} 
    brushR::Observable{Int32} 
    circ::Observable{Vector{Tuple{Int16, Int16}}} 

    # Magnetization
    M::Observable{Float32}

    # Updates per frame average
    upf::Observable{Int} 

    # Size of Image Window QML
    imgSize::Observable{Tuple{Int64, Int64}}

    # Analysis Is running
    analysisRunning::Observable{Bool}
end

Obs(;nlayers = 1, graphSize, initTemp = 1, initbrushR, initImg) = Obs(  
    # Active Layer
    Observable(Int32(1)),
    # Nlayers
    Observable(Int32(nlayers)),
    # Size of sim
    Observable(Int32(graphSize)),
    # Temperature
    Observable(Float32(initTemp)),
    # Brush
    Observable(Float32(0.)),
    # BrushR
    Observable( Int32(initbrushR) ),
    # Circ
    Observable(getOrdCirc(Int32(initbrushR))),
    # M
    Observable(Float32(0.0)),
    # UPF
    Observable(0),
    # Image Size
    Observable(size(initImg)),
    # Analysis Running
    Observable(false)
)

# Register QML observables to property map of sim
function register(sim, obs::Obs)
    for name in fieldnames(typeof(obs))
        sim.pmap[string(name)] = getproperty(obs, name)
    end
end


