include("IsingParams.jl")
include("Obs.jl")
# include("Processes.jl")
include("Pausing.jl")
include("ProcessList.jl")
include("IsingSim.jl")

# include("Algorithms/Algorithms.jl")
include("Process.jl")
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
    args...
    )
    createsimfunc = () -> IsingSim(len, wid; precision, periodic, type, weighted, weights, initTemp, colorscheme, args...)
    _assign_or_createsim(createsimfunc, register_sim)
    g = simulation[].gs[1]
    _simulate(g; start, gui, args...)
    return g
end

function simulate(g::IsingGraph{T}; start = true, gui = true, precision = T, initTemp = one(precision), colorscheme = ColorSchemes.viridis, register_sim = true, args...) where T
    createsimfunc = () -> IsingSim(g; start, initTemp, colorscheme)
    _assign_or_createsim(createsimfunc, register_sim; args...)
    _simulate(g; start, gui, args...)
    return g
end

function simulate(filename::String; start = true, register_sim = true, args...)
    createsimfunc = () -> IsingSim(filename; args...)
    restarted = _assign_or_createsim(createsimfunc, register_sim; args...)
    __simulate(_sim.gs[1]; start = restarted && start, gui = restarted && get(args, :gui, true), args...)
    return g
end

# Why use register sim?
"""
Pass in the appropraite IsingSim constructor
"""
function _assign_or_createsim(create_sim_func, noinput = true, register_sim = true; overwrite = false, args...)
    if isnothing(simulation) && noinput
        simulation[] = create_sim_func()
    elseif register_sim #If there is already a sim and we want to register a new one
        println("Simulation already active, create a new one and overwrite it? [y/n]")
        while true
            if overwrite
                s = 'y'
            else
                s = read(stdin, Char)
            end
            println("Character entered: $s")
            if s == 'y'
                closeinterface()
                reset!(simulation)
                simulation[] = create_sim_func()
                return true
            elseif s == 'n'
                return false
            else
                println("Please enter y or n")
            end
        end
    end
end

getgraph() = gs(simulation[])[1]

function _simulate(g; run = true, start = true, gui = true, args...)
    if start
        quit.(processes(g))
        createProcess(g; run)
    end
    if gui
        _interface(g; args...)
    end
end

interface(g; overwrite = true) = simulate(g; start = false, gui = true, overwrite)

export simulate, getgraph, interface



