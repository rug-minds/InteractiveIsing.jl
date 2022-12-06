export IsingSim

""" 
Simulation struct
"""
mutable struct IsingSim
    #Graph Layers
    const layers::Vector{IsingGraph}

    # Property map for qml
    const pmap::JuliaPropertyMap

    const M_array::Ref{Vector{Float64}}

    # Image of graph
    const img::Base.RefValue{Matrix{RGB{Float64}}}

    # Thread Locking
    const updatingUpf::Ref{Bool}
    const updatingMag::Ref{Bool} 
    const updatingImg::Ref{Bool} 

    # Simulation Parameters
    params::IsingParams
    # Observables
    obs::Obs


    #= Initializer =#
    function IsingSim(;
            continuous = false,
            graphSize = 512,
            weighted = true,
            weightFunc = defaultIsingWF,
            initTemp = 1.,
            start = false
        );
        
        g = IsingGraph(
            graphSize,
            continuous = continuous, 
            weighted = weighted,
            weightFunc = weighted ? weightFunc : defaultIsingWF
        )
        
        initImg = gToImg(g)
        initbrushR= round(graphSize/10)

        obs = Obs(;graphSize, initTemp, initbrushR, initImg)

        sim = new(
            # Layers
            [g],
            # Property map
            JuliaPropertyMap(),
            zeros(Real,60),
            img,
            Ref(false),
            Ref(false),
            Ref(false),
            IsingParams(),
            obs
        )

        # Register observables
        register(sim, obs)

        # Initialize image
        sim.img[] = initImg

        if start
            s()
        end
        return sim


    end
    #= END Initializer =# 
end


"""
Start the simulation and interface
Non-interface mode: WIP
"""
function (sim::IsingSim)(start = true; async = true)
    if start
        if Threads.nthreads() < 4
            error("Please enable multithreading to use the interface. For documentation, see github page")
        else
            startSim(sim; async)
        end
    end
    return sim.layers[1];
end

