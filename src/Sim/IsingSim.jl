export IsingSim

""" 
Simulation struct
"""
mutable struct IsingSim
    const gs::Vector{IsingGraph}
    #Graph Layers
    const layers::Vector{Vector{IsingLayer}}

    layeridx::Int32

    # Property map for qml
    const pmap::JuliaPropertyMap

    const M_array::Ref{Vector{Float64}}

    # Image of graph
    const img::Base.RefValue{Matrix{RGB{Float64}}}

    # Thread Locking
    const updatingUpf::Ref{Bool}
    const updatingMag::Ref{Bool} 
    const updatingImg::Ref{Bool} 

    #
    processes::Processes
    # Simulation Parameters
    params::IsingParams
    # Observables
    obs::Obs

    memory::Dict

    #= Initializer =#
    function IsingSim(
            len = 500,
            wid = 500;
            continuous = false,
            weighted = true,
            weightFunc = defaultIsingWF,
            initTemp = 1.,
            start = false,
            colorscheme = ColorSchemes.viridis
        );
        
        g = IsingGraph(
            len,
            wid,
            continuous = continuous, 
            weighted = weighted,
            weightFunc = weighted ? weightFunc : defaultIsingWF
        )

        newlayer = IsingLayer(g,1,len,wid)

        initbrushR= round(min(len,wid)/10)

        initImg = gToImg(newlayer; colorscheme)

        obs = Obs(;length = glength(newlayer), width = gwidth(newlayer), initTemp, initbrushR, initImg)
        
        sim = new(
            # Graphs
            [g],
            # Layers
            [[newlayer]],
            # Layer idx
            1,
            # Property map
            JuliaPropertyMap(),
            zeros(Real,60),
            # Image from module
            img,
            Ref(false),
            Ref(false),
            Ref(false),
            Processes(4),
            IsingParams(;initbrushR, colorscheme),
            obs,
            # memory
            Dict()
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
    return gs(sim)[1];
end

