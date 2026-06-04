using Dates
using Random
using SparseArrays
using Statistics

const DIAG_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(DIAG_ROOT, "mnist_local_manager_grid.jl"))

const DIAG_LOG = joinpath(@__DIR__, "r8_collapse_diagnostic.log")

"""Append one timestamped diagnostic line to stdout and the local log file."""
function diag_log(message::S; kwargs...) where {S<:AbstractString}
    payload = isempty(kwargs) ? "" : " " * join(("$(k)=$(v)" for (k, v) in pairs(kwargs)), " ")
    line = "[$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))] $message$payload"
    println(line)
    open(DIAG_LOG, "a") do io
        println(io, line)
    end
    flush(stdout)
    return nothing
end

"""Return compact scalar summaries for one vector slice."""
function vector_summary(v::V) where {V<:AbstractVector}
    return (;
        n = length(v),
        min = minimum(v),
        max = maximum(v),
        mean = mean(v),
        absmean = mean(abs, v),
        norm = sqrt(sum(abs2, v)),
        positives = count(>(zero(eltype(v))), v),
        negatives = count(<(zero(eltype(v))), v),
        zeros = count(==(zero(eltype(v))), v),
    )
end

"""Return class prediction data from an output-state vector."""
function output_summary(model::M, y::Y, state::S) where {M<:LocalMNISTModel,Y<:AbstractVector,S<:AbstractVector}
    output = @view state[model.output_idxs]
    scores = class_scores(output, model.config.output_replicas)
    target_scores = class_scores(y, model.config.output_replicas)
    return (;
        pred = argmax(scores) - 1,
        truth = argmax(target_scores) - 1,
        scores = collect(scores),
        output_sum = sum(output),
        output_pos = count(>(0f0), output),
        output_neg = count(<(0f0), output),
        loss = sum(abs2, y .- output) / 2,
    )
end

"""Summarize gradient magnitudes by local edge block."""
function gradient_summary(model::M, gradient::G) where {M<:LocalMNISTModel,G<:NamedTuple}
    kinds = model.edge_layout.kind
    labels = Dict(
        EDGE_INPUT_HIDDEN => "input_hidden",
        EDGE_HIDDEN_HIDDEN => "hidden_hidden",
        EDGE_HIDDEN_OUTPUT => "hidden_output",
        EDGE_HIDDEN1_INTERNAL => "hidden1_internal",
        EDGE_HIDDEN2_INTERNAL => "hidden2_internal",
        EDGE_OUTPUT_INTERNAL => "output_internal",
    )
    rows = NamedTuple[]
    for kind in sort(collect(keys(labels)))
        ptrs = findall(==(kind), kinds)
        isempty(ptrs) && continue
        vals = @view gradient.w[ptrs]
        push!(rows, (; block = labels[kind], n = length(ptrs), absmean = mean(abs, vals), maxabs = maximum(abs, vals), norm = sqrt(sum(abs2, vals))))
    end
    push!(rows, (; block = "bias_hidden1", n = length(model.hidden1_idxs), absmean = mean(abs, @view gradient.b[model.hidden1_idxs]), maxabs = maximum(abs, @view gradient.b[model.hidden1_idxs]), norm = sqrt(sum(abs2, @view gradient.b[model.hidden1_idxs]))))
    push!(rows, (; block = "bias_hidden2", n = length(model.hidden2_idxs), absmean = mean(abs, @view gradient.b[model.hidden2_idxs]), maxabs = maximum(abs, @view gradient.b[model.hidden2_idxs]), norm = sqrt(sum(abs2, @view gradient.b[model.hidden2_idxs]))))
    push!(rows, (; block = "bias_output", n = length(model.output_idxs), absmean = mean(abs, @view gradient.b[model.output_idxs]), maxabs = maximum(abs, @view gradient.b[model.output_idxs]), norm = sqrt(sum(abs2, @view gradient.b[model.output_idxs]))))
    return rows
end

"""Run one r=8 sample through the current Process path and print collapse clues."""
function main()
    isfile(DIAG_LOG) && rm(DIAG_LOG)
    config = LocalMNISTManagerConfig(;
        name = "r8_collapse_diagnostic",
        workers = 1,
        epochs = 1,
        batchsize = 1,
        train_per_class = 1,
        test_per_class = 1,
        local_radius = 8,
        free_sweeps = parse(Int, get(ENV, "ISING_R8_DIAG_SWEEPS", "25")),
        nudge_sweeps = parse(Int, get(ENV, "ISING_R8_DIAG_SWEEPS", "25")),
        β = parse(PMNIST_FT, get(ENV, "ISING_R8_DIAG_BETA", "5.0")),
        progress = false,
        progress_bar = false,
        outdir = @__DIR__,
    )
    diag_log("config"; radius = config.local_radius, sweeps = config.free_sweeps, beta = config.β, seed = config.seed)

    source = init_model(config, config.seed)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    x = copy(view(xtrain, :, 1))
    y = copy(view(ytrain, :, 1))
    diag_log("data"; x = vector_summary(x), y = vector_summary(y), truth = argmax(class_scores(y, config.output_replicas)) - 1)

    # Inspect field installation before the Process path mutates any state.
    field_model = worker_model(source, 1)
    base_bias = base_magfield(source.graph).b
    sample_buffer = zeros(PMNIST_FT, length(base_bias))
    install_sample_bias!(field_model, x, base_bias, sample_buffer)
    free_b = copy(base_magfield(field_model.graph).b)
    install_nudged_sample_bias!(field_model, x, y, base_bias, sample_buffer)
    nudged_b = copy(base_magfield(field_model.graph).b)
    diag_log(
        "field install";
        source_worker_shared_J = II.adj(source.graph) === II.adj(field_model.graph),
        worker_field_is_source_field = base_magfield(field_model.graph).b === base_magfield(source.graph).b,
        free_energy_graph = graph_energy(field_model.graph, field_model.graph.hamiltonian),
        free_field = vector_summary(free_b),
        free_h1_field = vector_summary(@view free_b[field_model.hidden1_idxs]),
        free_output_field = vector_summary(@view free_b[field_model.output_idxs]),
        nudged_output_field = vector_summary(@view nudged_b[field_model.output_idxs]),
    )

    dynamics_algorithm = mnist_dynamics_algorithm()
    worker_algorithm = StatefulAlgorithms.resolve(contrastive_worker_algorithm(deepcopy(dynamics_algorithm), config, length(II.state(source.graph))))
    worker = local_worker(source, 1, worker_algorithm)
    ctx = worker_context(worker)
    ctx.x .= x
    ctx.y .= y
    StatefulAlgorithms.reset!(worker)
    t_run = @elapsed begin
        run(worker)
        wait(worker)
    end
    diag_log("process sample finished"; seconds = round(t_run; digits = 4), nsamples = ctx.nsamples[], ncorrect = ctx.ncorrect[], nskipped = ctx.nskipped[], loss = ctx.total_loss[])
    diag_log("free output"; output_summary(ctx.mnist_model, y, ctx.free_state)...)
    diag_log("nudged output"; output_summary(ctx.mnist_model, y, ctx.nudged_state)...)
    diag_log(
        "state summaries";
        free_h1 = vector_summary(@view ctx.free_state[ctx.mnist_model.hidden1_idxs]),
        free_h2 = vector_summary(@view ctx.free_state[ctx.mnist_model.hidden2_idxs]),
        nudged_h1 = vector_summary(@view ctx.nudged_state[ctx.mnist_model.hidden1_idxs]),
        nudged_h2 = vector_summary(@view ctx.nudged_state[ctx.mnist_model.hidden2_idxs]),
    )
    for row in gradient_summary(ctx.mnist_model, ctx.gradient)
        diag_log("gradient"; row...)
    end

    manager = local_manager(source)
    try
        stats = run_minibatch!(manager, [LocalMNISTJob(x, y)]; log_progress = false)
        diag_log("one-sample optimizer step"; stats..., update_idx = manager.state.update_idx[])
        diag_log(
            "params after one update";
            w = vector_summary(manager.state.params[].w),
            b_hidden1 = vector_summary(@view manager.state.params[].b[source.hidden1_idxs]),
            b_hidden2 = vector_summary(@view manager.state.params[].b[source.hidden2_idxs]),
            b_output = vector_summary(@view manager.state.params[].b[source.output_idxs]),
        )
    finally
        close(manager)
    end
    diag_log("done"; log = DIAG_LOG)
    return nothing
end

main()
