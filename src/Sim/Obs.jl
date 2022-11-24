struct Obs
    # Active Layer
    activeLayer::Observable{Int32}
    # length/width of graph
    gSize::Observable{Int32}

    # Temperature Observable
    TIs::Observable{Float32}

    # For drawing to simulation
    brush::Observable{Float32} 
    brushR::Observable{Int32} 
    circ::Observable 

    # Magnetization
    M::Observable{Float32}

    # Updates per frame average
    upf::Observable{Int} 

    # For Branching Simulation
    shouldRun::Observable{Bool} 

    # Size of Image Window QML
    imgSize::Observable

    # Analysis Is running
    analysisRunning::Observable{Bool}
end

Obs(;graphSize, initTemp, initbrushR, initImg) = Obs(  
    # Active Layer
    Observable(Int32(1)),
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
    #Should Run
    Observable(true),
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


