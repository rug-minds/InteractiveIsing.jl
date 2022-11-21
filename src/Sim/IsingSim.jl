export IsingSim

""" 
Simulation struct
"""
mutable struct IsingSim
    # Graph
    const g::IsingGraph
    # Property map for qml
    const pmap::JuliaPropertyMap
    
    # length/width of graph
    const gSize::Observable{Int32}

    # Temperature Observable
    const TIs::Observable{Float32}

    # For drawing to simulation
    const brush::Observable{Float32} 
    const brushR::Observable{Int32} 
    const circ::Observable 

    # Magnetization
    const M::Observable{Float32} 
    const M_array::Ref{Vector{Real}}

    const analysisRunning::Observable{Bool} 
    
    # For tracking updates
    updates::Int
    # Updates per frame average
    const upf::Observable{Int} 

    # Image of graph
    const img::Base.RefValue{Matrix{RGB{Float64}}}
    const imgSize::Observable

    # Thread Locking
    const updatingUpf::Ref{Bool}
    const updatingMag::Ref{Bool} 
    const updatingImg::Ref{Bool} 

    # For Branching Simulation
    const shouldRun::Observable{Bool} 
    isRunning::Bool

    # Is sim already started?
    started::Bool


    #= Initializer =#
    function IsingSim(;
            continuous = false,
            graphSize = 512,
            weighted = true,
            weightFunc = defaultIsingWF,
            initTemp = 1.,
            start = false
        );

        type = continuous ? Float32 : Int8

        g = IsingGraph(
            graphSize,
            continuous = continuous, 
            weighted = weighted,
            weightFunc = weighted ? weightFunc : defaultIsingWF
        )
        
        initImg = gToImg(g)
        initbrushR= round(graphSize/10)

        sim = new(
            # Graph
            g,
            # Property map
            JuliaPropertyMap(),
            # Size of sim
            Observable(Int32(graphSize)),
            Observable(Float32(initTemp)),
            Observable(Float32(0.)),
            Observable( Int32(initbrushR) ),
            Observable(getOrdCirc(Int32(initbrushR))),
            Observable(Float32(0.0)),
            zeros(Real,60),
            Observable(false),
            0,
            Observable(0),
            img,
            Observable(size(initImg)),
            Ref(false),
            Ref(false),
            Ref(false),
            Observable(true),
            true,
            false
        )

        # Initialize image
        sim.img[] = initImg

        # Initializing propertymap
        sim.pmap["imgSize"] = sim.imgSize
        sim.pmap["shouldRun"] = sim.shouldRun
        sim.pmap["TIs"] = sim.TIs
        sim.pmap["brush"] = sim.brush
        sim.pmap["brushR"] = sim.brushR
        sim.pmap["circ"] = sim.circ 
        sim.pmap["M"] = sim.M
        sim.pmap["analysisRunning"] = sim.analysisRunning
        sim.pmap["upf"] = sim.upf
        sim.pmap["gSize"] = sim.gSize


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
    return sim.g;
end