"""
    _flat_process_algos(algo)

Return the algorithm nodes that should be scanned for Ising Monte Carlo
algorithms. Loop algorithms are flattened; non-loop algorithms are treated as a
single node.
"""
@inline _flat_process_algos(algo::Processes.AbstractLoopAlgorithm) = Processes.flat_funcs(algo)
@inline _flat_process_algos(algo) = (algo,)

"""
    _is_ising_mc_target(algo)

Return whether `algo` is an Ising Monte Carlo algorithm that needs the current
graph injected as its `model` input.
"""
@inline _is_ising_mc_target(algo) = Processes.algotype(algo) <: IsingMCAlgorithm

"""
    _collect_ising_mc_targets(func)

Collect the Ising Monte Carlo algorithm targets found inside `func`. The result
is a tuple so it can be passed through the process input construction path
without introducing vector storage.
"""
function _collect_ising_mc_targets(func)
    targets = ()
    for algo in _flat_process_algos(func)
        if _is_ising_mc_target(algo)
            targets = (targets..., algo)
        end
    end
    return targets
end

"""
    _mc_model_inits(func, g)

Create one `Processes.Init` per Ising Monte Carlo algorithm in `func`, assigning
`g` as that algorithm's `model`. User inputs are deliberately not inspected or
merged here; `createProcess` splats them into `Process` unchanged after these
graph-model inputs.
"""
function _mc_model_inits(func::F, g::G) where {F,G<:IsingGraph}
    targets = _collect_ising_mc_targets(func)
    graph_inputs = ()
    for target in targets
        graph_inputs = (graph_inputs..., Processes.Init(target, model = g))
    end
    return graph_inputs
end

"""
    createProcess(g::IsingGraph, func=nothing, inputs...; allow_multiple=false, kwargs...)

Create and run a process attached to `g`.

By default, existing processes on `g` are closed and removed before the new
process is launched, so repeated calls replace the active simulation. Pass
`allow_multiple=true` to keep existing processes and append the new one, which
matches the previous behavior. When no `lifetime`, `repeats`, or `repeat` is
given, the launched graph process runs indefinitely until it is closed. Pass
`looptype = Processes.RuntimeGenerated()` to select the process loop backend
used by the initial `run(process; ...)` call.
"""
function createProcess(
    g::IsingGraph,
    func = nothing,
    inputs...;
    dynamics = g.default_algorithm,
    lifetime = nothing,
    repeats = nothing,
    repeat = nothing,
    allow_multiple = false,
    looptype::LT = Processes.sys_looptype,
    args...,
) where {LT<:Processes.FunctionType}
    if isnothing(func)
        # func = get(args, :algorithm, g.default_algorithm)
        func = g.default_algorithm
    end

    if !isnothing(lifetime) && !(lifetime isa Processes.Lifetime)
        isnothing(repeats) || error("Pass either `repeats = ...` or numeric `lifetime = ...`, not both.")
        repeats = lifetime
        lifetime = nothing
    end

    if isnothing(lifetime) && isnothing(repeats) && isnothing(repeat)
        # Interactive graph processes should keep running unless the caller
        # provides an explicit stopping condition.
        lifetime = Processes.Indefinite()
    end

    if !allow_multiple
        Processes.close(g)
    end
    
    func = deepcopy(func)
    graph_inputs = _mc_model_inits(func, g)
    # process = Process(func, Init(DestructureInput(), structure = g); lifetime)
    process = Process(func, graph_inputs..., inputs...; lifetime, repeats, repeat)
    
    ps = processes(g)
    push!(ps, process)
    run(process; looptype)

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
