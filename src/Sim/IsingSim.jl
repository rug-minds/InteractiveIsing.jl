export IsingSim

""" 
Simulation struct
"""
mutable struct IsingSim
    # Graphs
    const gs::Vector{IG} where IG <: IsingGraph
    # Property map used in qml
    # const pmap::JuliaPropertyMap
    const pmap::Dict{String,Any}

    const M_array::Ref{CircularBuffer{Float64}}

    # Image of graph
    # const img::Ref{Matrix{RGB{Float64}}}

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
            len::Integer = 500,
            wid::Integer = 500;
            periodic = nothing,
            continuous = false,
            weighted = true,
            weights = nothing,
            initTemp = 1f0,
            colorscheme = ColorSchemes.viridis
        );
        

        initbrushR= round(min(len,wid)/10)

        sim = new(
            IsingGraph[],
            Dict{String,Any}(),
            CircularBuffer{Float64}(60),
            Ref(false),
            Ref(false),
            Ref(false),
            Processes(4),
            Timer[],
            IsingParams(;initbrushR, colorscheme),
            # memory
            Dict(),
        )
        
        sim.obs = Obs(len, wid, initbrushR, initTemp)

        g = IsingGraph(
            sim,
            len,
            wid;
            periodic,
            continuous,
            weighted,
            weights
        )

        initSim!(sim, g, len, wid, initbrushR, initTemp) 

        return sim

        
    end

    function IsingSim(g::IsingGraph; start = false, initTemp = 1f0, colorscheme = ColorSchemes.viridis)
        len = glength(g[1])
        wid = gwidth(g[1])

        initbrushR= round(min(glength(g[1]),gwidth(g[1]))/10)

        sim = new(
            # Graphs
            IsingGraph[],
            # Property map
            # JuliaPropertyMap(),
            Dict{String,Any}(),
            # M_array
            CircularBuffer{Float64}(60),
            # Image from module
            # img,
            Ref(false),
            Ref(false),
            Ref(false),
            Processes(4),
            Timer[],
            IsingParams(;initbrushR, colorscheme),
            # memory
            Dict(),
        )

        initSim!(sim, g, len, wid, initbrushR, initTemp)

        return sim

    end
    #= END of Initializer =# 
end 

# TODO: Is this still needed?
function destructor(sim::IsingSim)
    quit.(sim.processes)
    close.(sim.timers)
    destructor.(gs(sim))
    return nothing
end

function initSim!(sim, g, len, wid, initbrushR, initTemp)
    push!(gs(sim), g)
    sim.obs = Obs(len, wid, initbrushR, initTemp)
    sim.obs.layerName[] = name(g[1])
    nlayers(sim)[] = length(layers(g))
    g.sim = sim
    # Temperature Observable
    Temp(sim)[] = temp(g)

    on(Temp(sim), weak = true) do val
        temp(g, val)
    end

    on(brushR(sim), weak = true) do val
        circ(sim, getOrdCirc(val))
    end

    # finalizer(destructor, sim)
end


"""
Open from file
"""
function IsingSim(graphfile::String; kwargs...)
    g = loadGraph(graphfile)
    return IsingSim(g; kwargs...)
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

# @inline image(sim::IsingSim) = sim.img
# export image

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
        return sim[graphindex][layeridx]
    end
end

export layer

@inline graph(sim::IsingSim, graphidx = 1) = gs(sim)[graphidx]
export graph

function newGraph!(sim, len, wid; periodic = nothing, continuous = false, weighted = true, weights = nothing)
    g = IsingGraph(
        sim,
        len,
        wid;
        periodic,
        continuous,
        weighted,
        weights
    )

    push!(gs(sim), g)

    return g
end

deleteGraph!(sim, graphidx) = deleteGraph!(gs(sim)[graphidx])

function resetGraph!(sim, graphidx)
    cont = continuous(gs(sim)[graphidx]) == ContinuousState() ? true : false
    

end

export currentLayer
@inline currentLayer(sim) = layer(sim,layerIdx(sim)[])

