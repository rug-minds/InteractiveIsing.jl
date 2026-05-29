export LayeredIsingGraphLayer, LayerContrastiveStep, sync_params!

"""
    LayeredIsingGraphLayer(graph; input_idxs, output_idxs, β = 0.1f0, kwargs...)

Lux layer wrapper for an `InteractiveIsing.IsingGraph`.

The layer owns the static architecture and prepares reusable `Processes`
objects in `initialstates`. A forward application writes Lux parameters into
the graph, writes the input into the graph through the standard learning input
path, runs the prepared forward process once, and returns the configured output
state. Contrastive training uses the same graph/context model through
`LayerContrastiveStep`.
"""
struct LayeredIsingGraphLayer{G,I,O,B,D,N,V} <: LuxCore.AbstractLuxLayer
    model_graph::G
    input_layer::I
    output_layer::O
    β::B
    fullsweeps::Int
    nunits::Int
    dynamics_algorithm::D
    nudged_dynamics_algorithm::N
    validation_algorithm::V
    relaxation_steps::Int
    free_relaxation_steps::Int
    nudged_relaxation_steps::Int
end

"""
    LayeredIsingGraphLayer(graph_init; input_idxs, output_idxs, β, fullsweeps, ...)

Construct a graph-backed Lux layer. `graph_init` can be either a graph or a
zero-argument graph constructor.
"""
function LayeredIsingGraphLayer(
    graph_init;
    input_idxs,
    output_idxs,
    β::Real = 0.1f0,
    fullsweeps::Integer = 50,
    dynamics_algorithm = Metropolis(),
    nudged_dynamics_algorithm = dynamics_algorithm,
    validation_algorithm = dynamics_algorithm,
    relaxation_steps::Union{Nothing,Integer} = nothing,
    free_relaxation_steps::Union{Nothing,Integer} = nothing,
    nudged_relaxation_steps::Union{Nothing,Integer} = nothing,
)
    graph = graph_init isa Function ? graph_init() : graph_init
    n_units = nstates(graph)
    n_relaxation_steps = isnothing(relaxation_steps) ? Int(fullsweeps) * n_units : Int(relaxation_steps)
    n_free_relaxation_steps = isnothing(free_relaxation_steps) ? n_relaxation_steps : Int(free_relaxation_steps)
    n_nudged_relaxation_steps = isnothing(nudged_relaxation_steps) ? n_relaxation_steps : Int(nudged_relaxation_steps)
    beta = convert(eltype(graph), β)

    return LayeredIsingGraphLayer(
        graph,
        input_idxs,
        output_idxs,
        beta,
        Int(fullsweeps),
        n_units,
        dynamics_algorithm,
        nudged_dynamics_algorithm,
        validation_algorithm,
        n_relaxation_steps,
        n_free_relaxation_steps,
        n_nudged_relaxation_steps,
    )
end

"""Return the learnable graph parameters exposed to Lux."""
function initialparameters(rng::AbstractRNG, layer::LayeredIsingGraphLayer)
    graph = layer.model_graph
    base = (;
        w = copy(SparseArrays.getnzval(adj(graph))),
        b = copy(getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)),
    )
    isnothing(hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)) && return base
    return merge(base, (; α = copy(diag(adj(graph)))))
end

"""Allocate a contrastive-gradient buffer matching one graph."""
function layer_gradient_buffer(graph::G) where {G}
    base = (;
        w = zeros(eltype(graph), length(SparseArrays.getnzval(adj(graph)))),
        b = zeros(eltype(graph), nstates(graph)),
    )
    isnothing(hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)) && return base
    return merge(base, (; α = zeros(eltype(graph), nstates(graph))))
end

"""Build the reusable forward process stored in the Lux layer state."""
function layer_forward_process(layer::L, graph::G) where {L<:LayeredIsingGraphLayer,G}
    algorithm = resolve(ForwardDynamics(layer; dynamics_algorithm = layer.validation_algorithm).algorithm)
    return Process(
        algorithm,
        Init(:_state;
            x = zeros(eltype(graph), length(layer.input_layer)),
            equilibrium_state = copy(state(graph)),
        ),
        Init(:dynamics, model = graph);
        repeat = 1,
    )
end

"""
    LayerContrastiveStep(layer)

Single-example contrastive EP step implemented as a `ProcessAlgorithm`.
The context owns the graph, input, target, state captures, sampler contexts,
and gradient buffers. Repeated calls only change those buffers and graph state.
"""
struct LayerContrastiveStep{D,N,T} <: ProcessAlgorithm
    dynamics_algorithm::D
    nudged_dynamics_algorithm::N
    β::T
    input_dim::Int
    output_dim::Int
    free_relaxation_steps::Int
    nudged_relaxation_steps::Int
end

"""Create a contrastive process algorithm from a Lux graph layer."""
function LayerContrastiveStep(layer::L) where {L<:LayeredIsingGraphLayer}
    return LayerContrastiveStep(
        deepcopy(layer.dynamics_algorithm),
        deepcopy(layer.nudged_dynamics_algorithm),
        layer.β,
        length(layer.input_layer),
        length(layer.output_layer),
        layer.free_relaxation_steps,
        layer.nudged_relaxation_steps,
    )
end

"""Run a prepared dynamics context for `nsteps` single-spin/process updates."""
function relax_context!(algorithm::A, context::C, nsteps::Integer) where {A,C}
    @inbounds for _ in 1:Int(nsteps)
        Processes.step!(algorithm, context)
    end
    return context
end

"""Create persistent graph, sample, capture, sampler, and buffer storage."""
function Processes.init(step::LayerContrastiveStep, context)
    model = context.model
    T = eltype(model)
    x = get(context, :x, zeros(T, step.input_dim))
    y = get(context, :y, zeros(T, step.output_dim))
    buffers = get(context, :buffers, layer_gradient_buffer(model))
    equilibrium_state = get(context, :equilibrium_state, copy(state(model)))
    plus_state = get(context, :plus_state, similar(equilibrium_state))
    minus_state = get(context, :minus_state, similar(equilibrium_state))
    free_context = Processes.init(step.dynamics_algorithm, (; model))
    nudged_context = Processes.init(step.nudged_dynamics_algorithm, (; model))
    return (; model, x, y, buffers, equilibrium_state, plus_state, minus_state, free_context, nudged_context)
end

"""Run free, positive-nudged, and negative-nudged phases for one sample."""
function Processes.step!(step::LayerContrastiveStep, context)
    model = context.model
    β = step.β

    resetstate!(model)
    apply_input(model, context.x)
    relax_context!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)
    context.equilibrium_state .= state(model)

    state(model) .= context.equilibrium_state
    apply_input(model, context.x)
    apply_targets(model, context.y)
    set_clamping_beta!(model, β)
    relax_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.plus_state .= state(model)

    state(model) .= context.equilibrium_state
    apply_input(model, context.x)
    apply_targets(model, context.y)
    set_clamping_beta!(model, -β)
    relax_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.minus_state .= state(model)

    set_clamping_beta!(model, zero(β))
    contrastive_gradient(model, context.plus_state, context.minus_state, β; buffers = context.buffers)
    return nothing
end

"""Finalize a contrastive step; all state is intentionally reusable."""
function Processes.cleanup(step::LayerContrastiveStep, context)
    return nothing
end

"""Build the reusable contrastive process stored in the Lux layer state."""
function layer_contrastive_process(layer::L, graph::G) where {L<:LayeredIsingGraphLayer,G}
    algorithm = :_state => LayerContrastiveStep(layer)
    return Process(
        algorithm,
        Init(:_state;
            model = graph,
            x = zeros(eltype(graph), length(layer.input_layer)),
            y = zeros(eltype(graph), length(layer.output_layer)),
            buffers = layer_gradient_buffer(graph),
            equilibrium_state = copy(state(graph)),
            plus_state = similar(state(graph)),
            minus_state = similar(state(graph)),
        );
        repeat = 1,
    )
end

"""Create the mutable Lux state for a graph layer."""
function initialstates(rng::AbstractRNG, layer::LayeredIsingGraphLayer)
    graph = deepcopy(layer.model_graph)
    return (;
        graph,
        forward_process = layer_forward_process(layer, graph),
        contrastive_process = layer_contrastive_process(layer, graph),
    )
end

"""
    sync_params!(graph, ps)

Write Lux parameter arrays into the graph adjacency, base magnetic field, and
optional quadratic local-potential diagonal.
"""
function sync_params!(graph::G, ps::P) where {G<:IsingGraph,P}
    SparseArrays.getnzval(adj(graph)) .= ps.w
    getparam(graph.hamiltonian, InteractiveIsing.MagField, :b) .= ps.b

    quadratic = hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)
    if isnothing(quadratic)
        hasproperty(ps, :α) && error("graph has no Quadratic local potential but parameters contain α")
    else
        hasproperty(ps, :α) || error("graph has Quadratic local potential but parameters have no α")
        diag(adj(graph)) .= ps.α
        InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.Quadratic, :lp) .= ps.α
    end
    return graph
end

"""Run one reusable Lux forward process and return the configured output view."""
function (layer::LayeredIsingGraphLayer)(x, ps, st)
    graph = st.graph
    sync_params!(graph, ps)

    process = st.forward_process
    context = Processes.context(process)
    context._state.x .= x

    Processes.reset!(process)
    run(process)
    wait(process)

    return graph_view(graph, layer.output_layer), st
end
