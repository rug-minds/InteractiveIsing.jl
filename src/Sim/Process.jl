@inline _graph_input_vars(g::IsingGraph) = (; isinggraph = g, structure = g, state = g)

@inline _collect_ising_mc_targets(::Any) = ()
@inline _collect_ising_mc_targets(algo::IsingMCAlgorithm) = (algo,)

@inline function _collect_ising_mc_targets(algo::Processes.AbstractIdentifiableAlgo)
    if Processes.algotype(algo) <: IsingMCAlgorithm
        return (algo,)
    end
    return _collect_ising_mc_targets(Processes.getalgos(algo))
end

@inline function _collect_ising_mc_targets(algo::Processes.LoopAlgorithm)
    return _collect_ising_mc_targets(Processes.getalgos(algo))
end

@inline function _collect_ising_mc_targets(algos::Tuple)
    targets = ()
    for algo in algos
        targets = (targets..., _collect_ising_mc_targets(algo)...)
    end
    return targets
end

function _dedupe_targets(targets::Tuple)
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

function _merge_graph_inputs(func, g::IsingGraph, inputs...)
    targets = _dedupe_targets(_collect_ising_mc_targets(func))
    isempty(targets) && return inputs

    used_inputs = falses(length(inputs))
    merged_inputs = Any[]

    for target in targets
        merged_vars = _graph_input_vars(g)
        for (idx, input) in enumerate(inputs)
            if input isa Processes.Input && Processes.match(getfield(input, :target_algo), target)
                used_inputs[idx] = true
                merged_vars = merge(merged_vars, getfield(input, :vars))
            end
        end
        push!(merged_inputs, Processes.Input(target, pairs(merged_vars)...))
    end

    for (idx, input) in enumerate(inputs)
        if used_inputs[idx]
            continue
        end
        push!(merged_inputs, input)
    end

    return tuple(merged_inputs...)
end

function createProcess(g::IsingGraph, func = nothing, inputs...; dynamics = g.default_algorithm, lifetime = nothing, args...)
    if isnothing(func)
        # func = get(args, :algorithm, g.default_algorithm)
        func = g.default_algorithm
    end
    
    func = deepcopy(func)
    process_inputs = _merge_graph_inputs(func, g, inputs...)
    # process = Process(func, Input(DestructureInput(), structure = g); lifetime)
    process = Process(func, process_inputs...; lifetime)
    
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
