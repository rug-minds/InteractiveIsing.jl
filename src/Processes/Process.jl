"""
    _flat_process_algos(algo)

Return the algorithm nodes that should be scanned for Ising Monte Carlo
algorithms. Loop algorithms are flattened; non-loop algorithms are treated as a
single node.
"""
@inline _flat_process_algos(algo::StatefulAlgorithms.AbstractLoopAlgorithm) = StatefulAlgorithms.flat_funcs(algo)
@inline _flat_process_algos(algo) = (algo,)

"""
    _is_graph_model_target(algo)

Return whether `algo` needs the current graph injected as its `model` input.
"""
@inline function _is_graph_model_target(algo)
    algotype = StatefulAlgorithms.algotype(algo)
    return algotype <: IsingMCAlgorithm || algotype <: AdiabaticOptimization
end

"""
    _is_ising_mc_target(algo)

Return whether `algo` is an Ising Monte Carlo algorithm target.
"""
@inline function _is_ising_mc_target(algo)
    return StatefulAlgorithms.algotype(algo) <: IsingMCAlgorithm
end

"""
    _collect_graph_model_targets(func)

Collect graph-model process targets found inside `func`. The result is a tuple
so it can be passed through process input construction without vector storage.
"""
function _collect_graph_model_targets(func)
    targets = ()
    for algo in _flat_process_algos(func)
        if _is_graph_model_target(algo)
            targets = (targets..., algo)
        end
    end
    return targets
end

"""
    _collect_ising_mc_targets(func)

Collect only Ising Monte Carlo targets found inside `func`.
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

Create one `StatefulAlgorithms.Init` per graph-model algorithm in `func`,
assigning `g` as that algorithm's `model`. User inputs are deliberately not
inspected or merged here; `createProcess` splats them into `Process` unchanged
after these graph-model inputs.
"""
function _mc_model_inits(func::F, g::G) where {F,G<:IsingGraph}
    targets = _collect_graph_model_targets(func)
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
    target isa Symbol && return target

    key = try
        StatefulAlgorithms.static_findkey(registry, target)
    catch
        nothing
    end
    !isnothing(key) && return key

    if target isa StatefulAlgorithms.AbstractMatcher
        entry = try
            StatefulAlgorithms.get_by_matcher(registry, target)
        catch
            nothing
        end
        isnothing(entry) || return StatefulAlgorithms.getkey(entry)
    end

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

@inline _interactive_addon_enabled(addon::Bool) = addon
@inline _interactive_addon_enabled(::Any) = true
@inline _interactive_addon_enabled(::Nothing) = false

@inline _legacy_interactive_temperature(addon) = addon === true

@inline _interactive_addon_specs(::Bool) = ()
@inline _interactive_addon_specs(::Nothing) = ()
@inline _interactive_addon_specs(spec::StatefulAlgorithms.Interactive) = (spec,)
@inline _interactive_addon_specs(specs::Tuple) = filter(spec -> spec isa StatefulAlgorithms.Interactive, specs)

"""
    _push_prepared_interactive_spec(prepared_algo, specs, spec)

Resolve one graph-addon `Interactive` spec against a prepared process and add
the variable names that are present in the target subcontext.
"""
function _push_prepared_interactive_spec(prepared_algo, specs::Specs, spec::StatefulAlgorithms.Interactive) where {Specs<:Tuple}
    target = StatefulAlgorithms.get_target(spec)
    for varname in StatefulAlgorithms.interactive_names(spec)
        data = _prepared_interactive_var_data(prepared_algo, target, varname)
        isnothing(data) && continue
        target_name, _ = data
        specs = _push_unique_interactive_spec(specs, target_name, varname)
    end
    return specs
end

"""
    _mc_interactive_specs(func, g)

Build graph-driven `StatefulAlgorithms.Interactive` lifecycle specs and matching
`StatefulAlgorithms.Override` values for designated interactive process variables.
"""
function _mc_interactive_specs(func::F, g::G) where {F,G<:IsingGraph}
    interactive_addon = get(g, :interactive, false)
    (_interactive_addon_enabled(interactive_addon) || !isempty(interactivevars(g))) || return (), ()

    graph_inputs = _mc_model_inits(func, g)
    prepared_algo = StatefulAlgorithms.init(
        StatefulAlgorithms.normalize_process_algo(deepcopy(func)),
        graph_inputs...;
        lifetime = StatefulAlgorithms.Indefinite(),
    )

    interactive_specs = ()
    overrides = ()

    if _legacy_interactive_temperature(interactive_addon)
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

    for spec in _interactive_addon_specs(interactive_addon)
        interactive_specs = _push_prepared_interactive_spec(prepared_algo, interactive_specs, spec)
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
        func = dynamics
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

"""
    createProcessManual(g, func, inputs...; kwargs...)

Create and run a process attached to `g` without injecting graph-model `Init`s or
graph-driven interactive specs. Every algorithm that needs initialization data
must receive it explicitly through `inputs`.
"""
function createProcessManual(g::IsingGraph, func = nothing, inputs...; dynamics = g.default_algorithm, lifetime = nothing, repeats = nothing, repeat = nothing, allow_multiple = false, args...)
    if isnothing(func)
        func = dynamics
    end

    if !any(input -> input isa StatefulAlgorithms.Init, inputs)
        throw(ArgumentError("createProcessManual requires explicit StatefulAlgorithms.Init inputs."))
    end

    if !isnothing(lifetime) && !(lifetime isa StatefulAlgorithms.Lifetime)
        isnothing(repeats) || error("Pass either `repeats = ...` or numeric `lifetime = ...`, not both.")
        repeats = lifetime
        lifetime = nothing
    end

    if isnothing(lifetime) && isnothing(repeats) && isnothing(repeat)
        lifetime = StatefulAlgorithms.Indefinite()
    end

    if !allow_multiple
        StatefulAlgorithms.close(g)
    end

    process = Process(deepcopy(func), inputs...; lifetime, repeats, repeat)

    ps = processes(g)
    push!(ps, process)
    run(process)

    return process
end

export createProcess, createProcessManual, createProcesses

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
