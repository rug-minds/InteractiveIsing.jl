"""
    _graph_input_vars(g)

Runtime inputs injected into Ising Monte Carlo algorithms when a graph-backed
`Process` is created. Only `model` is injected; routes inside the algorithm are
responsible for forwarding that graph to names like `isinggraph` if needed.
"""
@inline _graph_input_vars(g::IsingGraph) = (; model = g)

"""
    _flat_process_algos(algo)

Return the algorithm nodes that should be scanned for Ising Monte Carlo
algorithms. Loop algorithms are flattened; non-loop algorithms are treated as a
single node.
"""
@inline _flat_process_algos(algo::Processes.LoopAlgorithm) = Processes.flat_funcs(algo)
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
    _dedupe_targets(targets::T) where {T<:Tuple}

Remove duplicate process targets by their `Processes.match_by` identity while
preserving the first occurrence. This prevents injecting the same graph input
twice for algorithms that appear through multiple flattened views.
"""
function _dedupe_targets(targets::T) where {T<:Tuple}
    deduped = Any[]
    seen = Any[]
    for target in targets
        matcher = Processes.match_by(target)
        if any(==(matcher), seen)
            continue
        end
        push!(seen, matcher)
        push!(deduped, target)
    end
    return tuple(deduped...)
end

"""
    _merge_graph_inputs(func, g, inputs...)

Return `inputs` plus one `Processes.Init` per Ising Monte Carlo algorithm in
`func`, with `model = g` merged into that target's existing init values. Existing
user inputs for the same target are preserved and override the injected `model`.
"""
function _merge_graph_inputs(func, g::IsingGraph, inputs...)
    targets = _dedupe_targets(_collect_ising_mc_targets(func))
    isempty(targets) && return inputs

    used_inputs = falses(length(inputs))
    merged_inputs = Any[]

    for target in targets
        merged_vars = _graph_input_vars(g)
        for (idx, input) in enumerate(inputs)
            if input isa Processes.Init && Processes.match(Processes.get_ref(input), target)
                used_inputs[idx] = true
                merged_vars = merge(merged_vars, Processes.get_vars(input))
            end
        end
        push!(merged_inputs, Processes.Init(target, pairs(merged_vars)...))
    end

    for (idx, input) in enumerate(inputs)
        if used_inputs[idx]
            continue
        end
        push!(merged_inputs, input)
    end

    return tuple(merged_inputs...)
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
