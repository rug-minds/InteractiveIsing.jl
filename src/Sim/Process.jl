@inline _flat_process_algos(algo::Processes.AbstractLoopAlgorithm) = Processes.flat_funcs(algo)
@inline _flat_process_algos(algo) = (algo,)

@inline _is_ising_mc_target(algo) = Processes.algotype(algo) <: IsingMCAlgorithm

"""
    _merge_graph_inputs(func, g::IsingGraph, inputs...)

Supply `g` as `model` to every Ising Monte Carlo algorithm inside `func`.
"""
function _merge_graph_inputs(func, g::IsingGraph, inputs...)
    targets = filter(_is_ising_mc_target, _flat_process_algos(func))
    model_inputs = Tuple(Processes.Init(target, model = g) for target in targets)
    return (model_inputs..., inputs...)
end

"""
    createProcess(g::IsingGraph, func=nothing, inputs...; allow_multiple=false, kwargs...)

Create and run a process attached to `g`.

By default, existing processes on `g` are closed and removed before the new
process is launched, so repeated calls replace the active simulation. Pass
`allow_multiple=true` to keep existing processes and append the new one, which
matches the previous behavior.
"""
function createProcess(g::IsingGraph, func = nothing, inputs...; dynamics = g.default_algorithm, lifetime = nothing, repeats = nothing, repeat = nothing, allow_multiple = false, args...)
    if isnothing(func)
        # func = get(args, :algorithm, g.default_algorithm)
        func = g.default_algorithm
    end

    if !isnothing(lifetime) && !(lifetime isa Processes.Lifetime)
        isnothing(repeats) || error("Pass either `repeats = ...` or numeric `lifetime = ...`, not both.")
        repeats = lifetime
        lifetime = nothing
    end

    if !allow_multiple
        Processes.close(g)
    end
    
    func = deepcopy(func)
    process_inputs = _merge_graph_inputs(func, g, inputs...)
    # process = Process(func, Init(DestructureInput(), structure = g); lifetime)
    process = Process(func, process_inputs...; lifetime, repeats, repeat)
    
    ps = processes(g)
    push!(ps, process)
    run(process)

    return process
end

export createProcess, createProcesses

hasprocess(g::IsingGraph) = !isempty(processes(g))
doneprocesses(g::IsingGraph) = findall(p -> isdone(p), processes(g))
norunningprocesses(g::IsingGraph) = length(doneprocesses(g)) == length(processes(g))
lastprocess(g::IsingGraph) = processes(g)[end]
cleanprocesses(g::IsingGraph) = deleteat!(processes(g), doneprocesses(g))

"""
Wait until all processes are done.
"""
Base.wait(g::IsingGraph) = wait.(processes(g))
"""
Fetch last output
"""
Base.fetch(g::IsingGraph) = fetch(process(g))
export wait, fetch

Processes.getcontext(g::IsingGraph) = getcontext(process(g))
export getcontext
