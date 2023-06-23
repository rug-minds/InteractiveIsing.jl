export IsingSim

""" 
Simulation struct
"""
mutable struct IsingSim
    # Graphs
    const gs::Vector{IG} where IG <: IsingGraph
    # Property map used in qml
    const pmap::JuliaPropertyMap

    const M_array::Ref{Vector{Float64}}

    # Image of graph
    const img::Ref{Matrix{RGB{Float64}}}

    # Thread Locking
    const updatingUpf::Ref{Bool}
    const updatingMag::Ref{Bool} 
    const updatingImg::Ref{Bool}

    # Process being used
    processes::Processes
    # Timers
    timers::Vector{Timer}
    # Simulation Parameters
    const params::IsingParams
    
    memory::Dict

    # Observables
    obs::Obs


    #= Initializer =#
    function IsingSim(
            len = 500,
            wid = 500;
            periodic = nothing,
            continuous = false,
            weighted = true,
            weightfunc = nothing,
            initTemp = 1.,
            start = false,
            colorscheme = ColorSchemes.viridis
        );
        

        initbrushR= round(min(len,wid)/10)

        sim = new(
            # Graphs
            IsingGraph[],
            # Property map
            JuliaPropertyMap(),
            zeros(Real,60),
            # Image from module
            img,
            Ref(false),
            Ref(false),
            Ref(false),
            Processes(4),
            Timer[],
            IsingParams(;initbrushR, colorscheme),
            # memory
            Dict(),
        )

        g = IsingGraph(
            sim,
            len,
            wid;
            periodic,
            continuous,
            weighted,
            weightfunc
        )

        push!(gs(sim), g)

        # Initialize image
        if !isempty(layers(g))
            initImg = gToImg(layer(g,1); colorscheme)
        end

        #Observables
        sim.obs = Obs(;length = len, width = wid, initTemp, initbrushR, initImg)
        
        sim.img[] = initImg
        # Register observables
        register(sim, sim.obs)



        if start
            s()
        end

        return sim

        
    end
    #= END of Initializer =# 
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

@forward IsingSim Obs
@forward IsingSim IsingParams params
@setterGetter IsingSim img

@inline image(sim::IsingSim) = sim.img
export image

getindex(sim::IsingSim, idx) = gs(sim)[idx]

#get n-th layer, starting the counting from the first graph
function layer(sim, layeridx)
    graphindex = 1
    if layeridx > length(layers(sim[graphindex]))
        layeridx -= length(layers(sim[graphindex]))
        graphindex += 1

        if layeridx > length(layers(sim[graphindex]))
            error("Layer index out of bounds")
        end
    else
        return layers(sim[graphindex])[layeridx]
    end
end

export layer

@inline graph(sim, graphidx = 1) = gs(sim)[graphidx]
export graph

export currentLayer
@inline currentLayer(sim) = layer(sim,layerIdx(sim)[])

