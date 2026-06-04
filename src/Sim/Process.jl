"""
    _flat_process_algos(algo)

Return the algorithm nodes that should be scanned for Ising Monte Carlo
algorithms. Loop algorithms are flattened; non-loop algorithms are treated as a
single node.
"""
@inline _flat_process_algos(algo::StatefulAlgorithms.AbstractLoopAlgorithm) = StatefulAlgorithms.flat_funcs(algo)
@inline _flat_process_algos(algo) = (algo,)

"""
    _is_ising_mc_target(algo)

Return whether `algo` is an Ising Monte Carlo algorithm that needs the current
graph injected as its `model` input.
"""
@inline _is_ising_mc_target(algo) = StatefulAlgorithms.algotype(algo) <: IsingMCAlgorithm

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

Create one `StatefulAlgorithms.Init` per Ising Monte Carlo algorithm in `func`, assigning
`g` as that algorithm's `model`. User inputs are deliberately not inspected or
merged here; `createProcess` splats them into `Process` unchanged after these
graph-model inputs.
"""
function _mc_model_inits(func::F, g::G) where {F,G<:IsingGraph}
    targets = _collect_ising_mc_targets(func)
    graph_inputs = ()
    for target in targets
        graph_inputs = (graph_inputs..., StatefulAlgorithms.Init(target, model = g))
    end
    return graph_inputs
end

@inline _interactive_slot_value(value::Real) = value
@inline _interactive_slot_value(value::Base.RefValue{<:Real}) = value[]
@inline _interactive_slot_value(value::StatefulAlgorithms.InteractiveVar{<:Real}) = value[]
@inline _interactive_slot_value(value::StatefulAlgorithms.InteractiveVar{<:Base.RefValue{<:Real}}) = value[][]
@inline _interactive_slot_value(_) = nothing

"""
    _resolve_interactive_target_key(registry, target)

Resolve a graph-level interactive target to one concrete process namespace key.
Broad type targets such as `LocalLangevin` are matched by registry subtype and
must resolve to exactly one entry.
"""
function _resolve_interactive_target_key(registry::R, target) where {R<:StatefulAlgorithms.NameSpaceRegistry}
    key = try
        StatefulAlgorithms.static_findkey(registry, target)
    catch
        nothing
    end
    !isnothing(key) && return key

    target isa Type || return nothing
    matches = Base.findall(target, registry)
    isempty(matches) && return nothing
    length(matches) == 1 ||
        error("Interactive target $(target) is ambiguous in this process. Use a keyed algorithm instance instead.")
    return StatefulAlgorithms.getkey(only(matches))
end

function _prepared_interactive_var_data(prepared_algo, target, varname::Symbol)
    registry = StatefulAlgorithms.getregistry(prepared_algo)
    target_name = _resolve_interactive_target_key(registry, target)
    isnothing(target_name) && return nothing
    subcontexts = StatefulAlgorithms.get_subcontexts(StatefulAlgorithms.context(prepared_algo))
    hasproperty(subcontexts, target_name) || return nothing
    data = StatefulAlgorithms.getdata(getproperty(subcontexts, target_name))
    haskey(data, varname) || return nothing
    return target_name, getproperty(data, varname)
end

@inline function _push_unique_interactive_spec(specs::Specs, target, varname::Symbol) where {Specs<:Tuple}
    for spec in specs
        if isequal(StatefulAlgorithms.get_target(spec), target) && only(StatefulAlgorithms.interactive_names(spec)) === varname
            return specs
        end
    end
    return (specs..., StatefulAlgorithms.Interactive(target, varname))
end

"""
    _mc_interactive_specs(func, g)

Build graph-driven `StatefulAlgorithms.Interactive` lifecycle specs and matching
`StatefulAlgorithms.Override` values for designated interactive process variables.
"""
function _mc_interactive_specs(func::F, g::G) where {F,G<:IsingGraph}
    Bool(get(g, :interactive, false)) || !isempty(interactivevars(g)) || return (), ()

    graph_inputs = _mc_model_inits(func, g)
    prepared_algo = StatefulAlgorithms.init(
        StatefulAlgorithms.normalize_process_algo(deepcopy(func)),
        graph_inputs...;
        lifetime = StatefulAlgorithms.Indefinite(),
    )

    interactive_specs = ()
    overrides = ()

    if Bool(get(g, :interactive, false))
        for target in _collect_ising_mc_targets(func)
            for varname in (:T, :temp)
                data = _prepared_interactive_var_data(prepared_algo, target, varname)
                isnothing(data) && continue
                target_name, _ = data
                interactive_specs = _push_unique_interactive_spec(interactive_specs, target_name, varname)
                break
            end
        end
    end

    for spec in interactivevars(g)
        data = _prepared_interactive_var_data(prepared_algo, spec.target, spec.varname)
        isnothing(data) && continue

        target_name, current = data
        value = isnothing(spec.value) ? _interactive_slot_value(current) : spec.value
        interactive_specs = _push_unique_interactive_spec(interactive_specs, target_name, spec.varname)
        isnothing(value) || (overrides = (overrides..., StatefulAlgorithms.Override(target_name, spec.varname => value)))
    end

    return overrides, interactive_specs
end

"""
    createProcess(g::IsingGraph, func=nothing, inputs...; allow_multiple=false, kwargs...)

Create and run a process attached to `g`.

By default, existing processes on `g` are closed and removed before the new
process is launched, so repeated calls replace the active simulation. Pass
`allow_multiple=true` to keep existing processes and append the new one, which
matches the previous behavior. When no `lifetime`, `repeats`, or `repeat` is
given, the launched graph process runs indefinitely until it is closed.
"""
function createProcess(g::IsingGraph, func = nothing, inputs...; dynamics = g.default_algorithm, lifetime = nothing, repeats = nothing, repeat = nothing, allow_multiple = false, args...)
    if isnothing(func)
        # func = get(args, :algorithm, g.default_algorithm)
        func = g.default_algorithm
    end

    if !isnothing(lifetime) && !(lifetime isa StatefulAlgorithms.Lifetime)
        isnothing(repeats) || error("Pass either `repeats = ...` or numeric `lifetime = ...`, not both.")
        repeats = lifetime
        lifetime = nothing
    end

    if isnothing(lifetime) && isnothing(repeats) && isnothing(repeat)
        # Interactive graph processes should keep running unless the caller
        # provides an explicit stopping condition.
        lifetime = StatefulAlgorithms.Indefinite()
    end

    if !allow_multiple
        StatefulAlgorithms.close(g)
    end
    
    func = deepcopy(func)
    interactive_overrides, interactive_specs = _mc_interactive_specs(func, g)
    graph_inputs = (_mc_model_inits(func, g)..., interactive_overrides..., interactive_specs...)
    # process = Process(func, Init(DestructureInput(), structure = g); lifetime)
    process = Process(func, graph_inputs..., inputs...; lifetime, repeats, repeat)
    
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

StatefulAlgorithms.getcontext(g::IsingGraph) = getcontext(process(g))
export getcontext
