include("IsingParams.jl")
include("Obs.jl")
include("Processes.jl")
include("Pausing.jl")
include("IsingSim.jl")

include("Algorithms/Algorithms.jl")
include("Loop.jl")
include("timedFunctions.jl")
include("User.jl")

const simulation = UnRef(IsingSim, destructor)

function simulate(
    len::Integer = 500,
    wid::Integer = 500;
    precision = Float32,
    periodic = nothing,
    type = Continuous,
    weighted = true,
    weights = nothing,
    initTemp = one(precision),
    start = true,
    gui = true,
    colorscheme = ColorSchemes.viridis,
    #for precompilation, should otherwise be set to true always
    register_sim = true,
    kwargs...
    )
    createsimfunc = () -> IsingSim(len, wid; precision, periodic, type, weighted, weights, initTemp, colorscheme, kwargs...)
    _assign_or_createsim(createsimfunc, register_sim)
    g = simulation[].gs[1]
    _simulate(g; start, gui, kwargs...)
    return g
end

function simulate(g::IsingGraph{T}; start = true, gui = true, precision = T, initTemp = one(precision), colorscheme = ColorSchemes.viridis, register_sim = true, kwargs...) where T
    createsimfunc = () -> IsingSim(g; start, initTemp, colorscheme)
    println("Making the sim")
    _assign_or_createsim(createsimfunc, register_sim)
    println("Assigning g")
    println("Simulating")
    _simulate(g; start, gui, kwargs...)
    return g
end

function simulate(filename::String; start = true, register_sim = true, kwargs...)
    createsimfunc = () -> IsingSim(filename; kwargs...)
    _assign_or_createsim(createsimfunc, register_sim)
    __simulate(_sim.gs[1]; start, gui, kwargs...)
    return g
end

# Why use register sim?
"""
Pass in the appropraite IsingSim constructor
"""
function _assign_or_createsim(create_sim_func, noinput = true, register_sim = true)
    if isnothing(simulation) && noinput
        simulation[] = create_sim_func()
    elseif register_sim #If there is already a sim and we want to register a new one
        println("Simulation already active, create a new one and overwrite it? [y/n]")
        while true
            s = read(stdin, Char)
            println("Character entered: $s")
            if s == 'y'
                println("Overwriting old sim")
                reset!(simulation)
                closeinterface()
                simulation[] = create_sim_func()
                return
            elseif s == 'n' 
                println("Doing nothing")
                return create_sim_func()
            else
                println("Please enter y or n")
            end
        end
    end
end

getgraph() = gs(simulation[])[1]

function _simulate(g; start = true, gui = true, kwargs...)
    run = searchkey(kwargs, :run; fallback = true)
    if start
        createProcess(g; run)
    end
    if gui
        interface(g; kwargs...)
    end
end

export simulate, getgraph



