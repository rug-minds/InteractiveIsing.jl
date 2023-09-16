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
    weightfunc = nothing,
    initTemp = 1f0,
    start = true,
    gui = true,
    colorscheme = ColorSchemes.viridis
    )
    if isnothing(simulation)
        simulation[] = IsingSim(len, wid; periodic, continuous, weighted, weightfunc, initTemp, start, colorscheme)
    end
    g = simulation[].gs[1]
    _simulate(g; start, gui)
    return g
end

function (g::IsingGraph; start = true, giu = true, initTemp = 1f0, colorscheme = ColorSchemes.viridis)
    if isnothing(simulation)
        simulation[] = IsingSim(graph; start, initTemp, colorscheme)
    end
    g = simulation[].gs[1]
    _simulate(g; start, gui)
    return g
end

function simulate(filename::String; start = true, kwargs...)
    if isnothing(simulation)
        simulation[] = IsingSim(filename; kwargs...)
    end
    
    return g
end

getgraph() = gs(simulation[])[1]

function interface(g)
    createBaseFig(g, create_singleview)
end

function _simulate(g ;kwargs...)
    g = simulation[].gs[1]
    if kwargs[:start]
        createProcess(g)
    end
    if kwargs[:gui]
        interface(g)
    end
end

export simulate, getgraph



