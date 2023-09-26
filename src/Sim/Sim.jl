include("IsingParams.jl")
include("Obs.jl")
include("Processes.jl")
include("Pausing.jl")
include("IsingSim.jl")

# include("QML.jl")
include("Loop.jl")
include("timedFunctions.jl")
include("User.jl")

const simulation = UnRef(IsingSim)

function simulate(
    len::Integer = 500,
    wid::Integer = 500;
    periodic = nothing,
    continuous = false,
    weighted = true,
    weights = nothing,
    initTemp = 1f0,
    start = true,
    gui = true,
    colorscheme = ColorSchemes.viridis,
    #for precompilation, should otherwise be set to true always
    register_sim = true,
    kwargs...
    )
    _sim = IsingSim(len, wid; periodic, continuous, weighted, weights, initTemp, colorscheme)
    if isnothing(simulation) && register_sim
        simulation[] = _sim
    end
    g = _sim.gs[1]
    _simulate(g; start, gui, kwargs...)
    return g
end

function simulate(g::IsingGraph; start = true, giu = true, initTemp = 1f0, colorscheme = ColorSchemes.viridis, register_sim = true, kwargs...)
    _sim = IsingSim(graph; start, initTemp, colorscheme)
    if isnothing(simulation) && register_sim
        simulation[] = _sim
    end
    g = _sim.gs[1]
    _simulate(g; start, gui, kwargs...)
    return g
end

function simulate(filename::String; start = true, register_sim = true, kwargs...)
    if isnothing(simulation) && register_sim
        simulation[] = IsingSim(filename; kwargs...)
    end
    
    return g
end

getgraph() = gs(simulation[])[1]

function _simulate(g; kwargs...)
    if kwargs[:start]
        createProcess(g)
    end
    if kwargs[:gui]
        interface(g; kwargs...)
    end
end

function interface(g; kwargs...)
    if haskey(kwargs, :disp)
        createBaseFig(g, create_singleview, disp = kwargs[:disp])
    else
        createBaseFig(g, create_singleview)
    end

end

export simulate, getgraph



