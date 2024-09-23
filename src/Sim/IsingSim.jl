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
    processes::ProcessList
    # Timers
    timers::Dict{String,PTimer}
    # Simulation Parameters
    const params::IsingParams
    
    memory::Dict

    # Observables
    obs::Obs

    ml::UnRef{SimLayout}


    #= Initializer =#
    function IsingSim(
            len::Integer = 500,
            wid::Integer = 500;
            initTemp = 1f0,
            kwargs...
        );
        

        initbrushR= round(min(len,wid)/10)

        sim = new(
            IsingGraph[],
            Dict{String,Any}(),
            CircularBuffer{Float64}(60),
            Ref(false),
            Ref(false),
            Ref(false),
            ProcessList(4),
            Dict{String,Timer}(),
            IsingParams(;initbrushR, colorscheme),
            # memory
            Dict(),
        )
        
        sim.obs = Obs(len, wid, initbrushR, initTemp)

        g = IsingGraph(
            len,
            wid;
            sim,
            kwargs...
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
            ProcessList(4),
            Dict{String,Timer}(),
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
    close.(values(timers(sim)))
    empty!(timedFunctions)
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
    temp(sim)[] = temp(g)

    on(temp(sim)) do val
        temp(g, val)
    end

    on(brushR(sim), weak = true) do val
        circ(sim, getOrdCirc(val))
    end

    sim.ml = UnRef(SimLayout(Figure()))

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

function get_gidx(g::IsingGraph)
    for (idx,graph) in enumerate(gs(sim(g)))
        if graph === g
            return idx
        end
    end
    return nothing
end

@forwardfields IsingSim Obs obs temp
@forwardfields IsingSim IsingParams params
@setterGetter IsingSim img
getindex(sim::IsingSim, idx) = gs(sim)[idx]
temp(sim::IsingSim, val) = sim.obs.temp[] = val
temp(sim::IsingSim) = sim.obs.temp

starttimers(sim::IsingSim) = start.(values(timers(sim)))
closetimers(sim::IsingSim) = close.(values(timers(sim)))
export starttimers, closetimers


function layer(sim, layeridx)
    g = gs(sim)[1]
    # println("Layers ", length(layers(g)))
    return g[layeridx]
end

export layer

@inline graph(sim::IsingSim, graphidx = 1) = gs(sim)[graphidx]
export graph

function newGraph!(sim, len, wid; periodic = nothing, continuous = false, weighted = true, weights = nothing)
    g = IsingGraph(
        len,
        wid;
        sim,
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

