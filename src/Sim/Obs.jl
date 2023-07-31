struct Obs
    # Active Layer
    layerIdx::Observable{Int32}
    # Name of the selected layer
    layerName::Observable{String}
    #number of layers
    nlayers::Observable{Int32}
    # length/width of graph
    qmllength::Observable{Int32}
    
    qmlwidth::Observable{Int32}

    # Temperature Observable
    Temp::Observable{Float32}

    # For drawing to simulation
    brush::Observable{Float32} 
    brushR::Observable{Int32} 

    # Magnetization
    M::Observable{Float32}

    # Updates per frame average
    upf::Observable{Float32} 

    # Size of Image Window QML
    imgSize::Observable{Tuple{Int64, Int64}}

    # Analysis Is running
    analysisRunning::Observable{Bool}

    # Is Paused
    isPaused::Observable{Bool}
    
    # Run the timed functions
    runTimedFunctions::Observable{Bool}

end

Obs(;nlayers = 0, length, width, initTemp = 1, initbrushR, initName = "") = Obs(  
    # Active Layer
    Observable(Int32(1)),
    # Name of the selected layer
    Observable(initName),
    # Nlayers
    Observable(Int32(nlayers)),
    # length of screen
    Observable(Int32(length)),
    # width of screen
    Observable(Int32(width)),
    # Temperature
    Observable(Float32(initTemp)),
    # Brush
    Observable(Float32(0.)),
    # BrushR
    Observable( Int32(initbrushR) ),
    # M
    Observable(Float32(0.0)),
    # UPF
    Observable(0),
    # Image Size
    Observable(Int64.((length,width))),
    # Analysis Running
    Observable(false),
    # Is Paused
    Observable(false),
    # Run Timed Functions
    Observable(true)
)

# Register QML observables to property map of sim
function register(sim, obs::Obs)
    for name in fieldnames(typeof(obs))
        sim.pmap[string(name)] = getproperty(obs, name)
    end
end


