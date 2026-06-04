using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using MLDatasets
using Random
using Serialization
using Statistics

const II = IsingLearning.InteractiveIsing

const INPUT_DIM = 28^2
const NCLASSES = 10
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_PAPER_HIDDEN", "120"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_PAPER_OUTPUT_REPLICAS", "4"))
const TRAIN_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_PAPER_TRAIN_PER_CLASS", "100"))
const TEST_PER_CLASS = parse(Int, get(ENV, "ISING_MNIST_PAPER_TEST_PER_CLASS", "10"))
const EPOCHS = parse(Int, get(ENV, "ISING_MNIST_PAPER_EPOCHS", "20"))
const FREE_READS = parse(Int, get(ENV, "ISING_MNIST_PAPER_FREE_READS", "10"))
const NUDGE_READS = parse(Int, get(ENV, "ISING_MNIST_PAPER_NUDGE_READS", "10"))
const FREE_SWEEPS = parse(Int, get(ENV, "ISING_MNIST_PAPER_FREE_SWEEPS", "100"))
const NUDGE_SWEEPS = parse(Int, get(ENV, "ISING_MNIST_PAPER_NUDGE_SWEEPS", "100"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_PAPER_BETA", "5.0"))
const LR_W0 = parse(Float32, get(ENV, "ISING_MNIST_PAPER_LR_W0", "0.01"))
const LR_W1 = parse(Float32, get(ENV, "ISING_MNIST_PAPER_LR_W1", "0.01"))
const LR_B0 = parse(Float32, get(ENV, "ISING_MNIST_PAPER_LR_B0", "0.001"))
const LR_B1 = parse(Float32, get(ENV, "ISING_MNIST_PAPER_LR_B1", "0.001"))
const GAIN_W0 = parse(Float32, get(ENV, "ISING_MNIST_PAPER_GAIN_W0", "0.5"))
const GAIN_W1 = parse(Float32, get(ENV, "ISING_MNIST_PAPER_GAIN_W1", "0.25"))
const WEIGHT_CLIP = parse(Float32, get(ENV, "ISING_MNIST_PAPER_WEIGHT_CLIP", "1.0"))
const BIAS_CLIP = parse(Float32, get(ENV, "ISING_MNIST_PAPER_BIAS_CLIP", "1.0"))
const APPLIED_BIAS_CLIP = parse(Float32, get(ENV, "ISING_MNIST_PAPER_APPLIED_BIAS_CLIP", "4.0"))
const HOT_TEMP = parse(Float32, get(ENV, "ISING_MNIST_PAPER_HOT_TEMP", "5.0"))
const COLD_TEMP = parse(Float32, get(ENV, "ISING_MNIST_PAPER_COLD_TEMP", "0.01"))
const REVERSE_TEMP = parse(Float32, get(ENV, "ISING_MNIST_PAPER_REVERSE_TEMP", "1.0"))
const OUTDIR = get(ENV, "ISING_MNIST_PAPER_DIR", joinpath(@__DIR__, "runs", Dates.format(now(), "yyyymmdd_HHMMSS_paper_like")))
const LOAD_PATH = get(ENV, "ISING_MNIST_PAPER_LOAD", "")

mkpath(OUTDIR)

mutable struct PaperMLP{T<:AbstractFloat,G,W0,W1,B0,B1}
    graph::G
    weights_0::W0
    weights_1::W1
    bias_0::B0
    bias_1::B1
    hidden_idxs::Vector{Int}
    output_idxs::Vector{Int}
    rng::Random.MersenneTwister
end

"""
    append_row!(path, row)

Append a named tuple to a CSV file, creating the header when needed.
"""
function append_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
    hidden_shape(hidden)

Pick a display-friendly 2D shape for the hidden layer without changing the
number of units.
"""
function hidden_shape(hidden::Integer)
    side = floor(Int, sqrt(hidden))
    while side > 1 && hidden % side != 0
        side -= 1
    end
    return side, hidden ÷ side
end

"""
    balanced_mnist(split, per_class)

Load a balanced MNIST subset. Inputs are kept in `[0, 1]`, matching the paper's
`ToTensor` preprocessing, and targets are repeated `+1/-1` Ising labels.
"""
function balanced_mnist(split::Symbol, per_class::Integer)
    dataset = split === :train ? MNIST(split = :train) : MNIST(split = :test)
    images, labels = dataset[:]
    buckets = [Int[] for _ in 1:NCLASSES]
    for idx in eachindex(labels)
        push!(buckets[Int(labels[idx]) + 1], idx)
    end

    keep = Int[]
    sizehint!(keep, NCLASSES * Int(per_class))
    for digit in 1:NCLASSES
        append!(keep, @view buckets[digit][1:Int(per_class)])
    end

    x = Matrix{Float32}(undef, INPUT_DIM, length(keep))
    y = fill(-1f0, NCLASSES * OUTPUT_REPLICAS, length(keep))
    for (col, idx) in enumerate(keep)
        x[:, col] .= Float32.(reshape(images[:, :, idx], :))
        maximum(@view x[:, col]) > 1.5f0 && (x[:, col] ./= 255f0)
        label = Int(labels[idx]) + 1
        first_idx = (label - 1) * OUTPUT_REPLICAS + 1
        y[first_idx:(first_idx + OUTPUT_REPLICAS - 1), col] .= 1f0
    end
    return x, y
end

"""
    build_graph()

Construct the hidden-output Ising graph used by the paper-like MLP. The input
layer is not represented as spins; `x * W0` is applied as hidden bias instead.
"""
function build_graph()
    hidden_rows, hidden_cols = hidden_shape(HIDDEN)
    output_rows, output_cols = hidden_shape(NCLASSES * OUTPUT_REPLICAS)
    zero_wg = II.AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0f0)
    hidden_layer = II.Layer(
        hidden_rows,
        hidden_cols,
        II.StateSet(-1f0, 1f0),
        II.Discrete(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    output_layer = II.Layer(
        output_rows,
        output_cols,
        II.StateSet(-1f0, 1f0),
        II.Discrete(),
        II.Coords(0, hidden_cols + 2, 0);
        periodic = false,
    )
    bias = g -> II.filltype(Vector, 0f0, II.statelen(g))
    graph = II.IsingGraph(
        hidden_layer,
        zero_wg,
        output_layer,
        II.Bilinear() + II.MagField(b = bias);
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, COLD_TEMP)
    return graph
end

"""
    init_model(seed)

Initialize paper-style weights with uniform Xavier-like scaling and the gains
reported in the released reproduction command.
"""
function init_model(seed::Integer = 1)
    rng = Random.MersenneTwister(seed)
    graph = build_graph()
    output_dim = NCLASSES * OUTPUT_REPLICAS
    weights_0 = GAIN_W0 .* (2f0 .* rand(rng, Float32, INPUT_DIM, HIDDEN) .- 1f0) .* sqrt(1f0 / INPUT_DIM)
    weights_1 = GAIN_W1 .* (2f0 .* rand(rng, Float32, HIDDEN, output_dim) .- 1f0) .* sqrt(1f0 / HIDDEN)
    bias_0 = zeros(Float32, HIDDEN)
    bias_1 = zeros(Float32, output_dim)
    hidden_idxs = collect(II.layerrange(graph[1]))
    output_idxs = collect(II.layerrange(graph[2]))
    model = PaperMLP{Float32,typeof(graph),typeof(weights_0),typeof(weights_1),typeof(bias_0),typeof(bias_1)}(
        graph,
        weights_0,
        weights_1,
        bias_0,
        bias_1,
        hidden_idxs,
        output_idxs,
        rng,
    )
    sync_graph_readout!(model)
    return model
end

"""
    sync_graph_readout!(model)

    Copy the trainable hidden-output readout and output biases into the graph.
    The paper's `dimod` Ising energy is `h*s + J*s*s`; this package's
    Hamiltonian is `-b*s - J*s*s`, so graph parameters store the negative of
    the paper parameters.
"""
function sync_graph_readout!(model::M) where {M<:PaperMLP}
    A = II.adj(model.graph)
    @inbounds for hpos in axes(model.weights_1, 1)
        hidden_idx = model.hidden_idxs[hpos]
        for opos in axes(model.weights_1, 2)
            A[model.output_idxs[opos], hidden_idx] = -model.weights_1[hpos, opos]
        end
    end
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    b[model.output_idxs] .= .-model.bias_1
    return model
end

"""
    apply_sample_bias!(model, x; target=nothing, beta=0)

    Apply the paper's per-sample hidden bias `clip(b0 + x'W0)` and, when
    requested, the Ising nudge output bias `b1 - beta * target`. The installed
    graph fields are negated to compensate for this package's Hamiltonian sign.
"""
function apply_sample_bias!(
    model::M,
    x::X;
    target = nothing,
    beta::Real = 0,
) where {M<:PaperMLP,X<:AbstractVector}
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    hidden_bias = vec(transpose(x) * model.weights_0) .+ model.bias_0
    b[model.hidden_idxs] .= .-clamp.(hidden_bias, -APPLIED_BIAS_CLIP, APPLIED_BIAS_CLIP)
    if isnothing(target)
        b[model.output_idxs] .= .-clamp.(model.bias_1, -APPLIED_BIAS_CLIP, APPLIED_BIAS_CLIP)
    else
        b[model.output_idxs] .= .-clamp.(model.bias_1 .- Float32(beta) .* target, -APPLIED_BIAS_CLIP, APPLIED_BIAS_CLIP)
    end
    return model
end

"""
    randomize_state!(model)

Reset all hidden/output spins to random Ising states.
"""
function randomize_state!(model::M) where {M<:PaperMLP}
    s = II.state(model.graph)
    @inbounds for idx in eachindex(s)
        s[idx] = rand(model.rng, Bool) ? 1f0 : -1f0
    end
    return model
end

"""
    graph_energy(model)

    Evaluate the current paper-convention Ising energy for selecting the
    lowest-energy read. The graph stores negated parameters, so `-b_graph*s`
    reproduces the paper's `h*s` term.
"""
function graph_energy(model::M) where {M<:PaperMLP}
    s = II.state(model.graph)
    b = II.getparam(model.graph.hamiltonian, II.MagField, :b)
    energy = 0f0
    @inbounds for hpos in axes(model.weights_1, 1)
        hidden_state = s[model.hidden_idxs[hpos]]
        for opos in axes(model.weights_1, 2)
            energy += model.weights_1[hpos, opos] * hidden_state * s[model.output_idxs[opos]]
        end
    end
    @inbounds for idx in eachindex(s)
        energy -= b[idx] * s[idx]
    end
    return energy
end

"""
    metropolis_sweep!(algorithm, context, nactive)

Run one full active-spin sweep of the Metropolis dynamics.
"""
function metropolis_sweep!(algorithm::A, context::C, nactive::Integer) where {A,C}
    for _ in 1:Int(nactive)
        StatefulAlgorithms.step!(algorithm, context)
    end
    return context
end

"""
    anneal!(model, context, sweeps; reverse=false)

Run a simple temperature schedule. Free phases cool from `HOT_TEMP` to
`COLD_TEMP`; nudged phases start from the free state, bump to `REVERSE_TEMP`,
then cool again.
"""
function anneal!(model::M, context::C, sweeps::Integer; reverse::Bool = false) where {M<:PaperMLP,C}
    nactive = length(II.state(model.graph))
    total = max(Int(sweeps), 1)
    for sweep in 1:total
        progress = total == 1 ? 1f0 : Float32(sweep - 1) / Float32(total - 1)
        if reverse
            half = 0.5f0
            if progress <= half
                local_progress = progress / half
                T = COLD_TEMP + local_progress * (REVERSE_TEMP - COLD_TEMP)
            else
                local_progress = (progress - half) / half
                T = REVERSE_TEMP + local_progress * (COLD_TEMP - REVERSE_TEMP)
            end
        else
            T = HOT_TEMP * (COLD_TEMP / HOT_TEMP)^progress
        end
        II.temp!(model.graph, T)
        metropolis_sweep!(II.Metropolis(), context, nactive)
    end
    II.temp!(model.graph, COLD_TEMP)
    return model
end

"""
    sample_phase!(model, x; target=nothing, beta=0, reads, sweeps, initial_state=nothing, reverse=false)

Run several annealing reads and return the lowest-energy hidden/output state.
"""
function sample_phase!(
    model::M,
    x::X;
    target = nothing,
    beta::Real = 0,
    reads::Integer,
    sweeps::Integer,
    initial_state = nothing,
    reverse::Bool = false,
) where {M<:PaperMLP,X<:AbstractVector}
    context = StatefulAlgorithms.init(II.Metropolis(), (; model = model.graph))
    best_energy = Inf32
    best_state = copy(II.state(model.graph))
    for _ in 1:Int(reads)
        if isnothing(initial_state)
            randomize_state!(model)
        else
            II.state(model.graph) .= initial_state
        end
        apply_sample_bias!(model, x; target, beta)
        anneal!(model, context, sweeps; reverse)
        energy = Float32(graph_energy(model))
        if energy < best_energy
            best_energy = energy
            best_state .= II.state(model.graph)
        end
    end
    return best_state, best_energy
end

"""
    class_scores(output)

Average repeated output spins into one class score per digit.
"""
function class_scores(output::V) where {V<:AbstractVector}
    scores = zeros(Float32, NCLASSES)
    @inbounds for digit in 1:NCLASSES
        first_idx = (digit - 1) * OUTPUT_REPLICAS + 1
        scores[digit] = sum(view(output, first_idx:(first_idx + OUTPUT_REPLICAS - 1))) / OUTPUT_REPLICAS
    end
    return scores
end

"""
    train_one!(model, x, y)

Run one paper-like one-sided EP update for a single MNIST sample.
"""
function train_one!(model::M, x::X, y::Y) where {M<:PaperMLP,X<:AbstractVector,Y<:AbstractVector}
    free_state, _ = sample_phase!(model, x; reads = FREE_READS, sweeps = FREE_SWEEPS)
    free_hidden = copy(@view free_state[model.hidden_idxs])
    free_output = copy(@view free_state[model.output_idxs])
    if all(free_output .== y)
        return (loss = sum(abs2, y .- free_output) / 2, correct = argmax(class_scores(free_output)) == argmax(class_scores(y)), skipped = true)
    end

    nudged_state, _ = sample_phase!(
        model,
        x;
        target = y,
        beta = BETA,
        reads = NUDGE_READS,
        sweeps = NUDGE_SWEEPS,
        initial_state = free_state,
        reverse = true,
    )
    nudged_hidden = @view nudged_state[model.hidden_idxs]
    nudged_output = @view nudged_state[model.output_idxs]

    invβ = 1f0 / BETA
    hidden_delta = nudged_hidden .- free_hidden
    output_delta = nudged_output .- free_output

    model.weights_1 .+= LR_W1 .* (.-(nudged_hidden * transpose(nudged_output) .- free_hidden * transpose(free_output)) .* invβ)
    model.weights_0 .+= LR_W0 .* (.-(x * transpose(hidden_delta)) .* invβ)
    model.bias_1 .+= LR_B1 .* (.-output_delta .* invβ)
    model.bias_0 .+= LR_B0 .* (.-hidden_delta .* invβ)

    model.weights_1 .= clamp.(model.weights_1, -WEIGHT_CLIP, WEIGHT_CLIP)
    model.weights_0 .= clamp.(model.weights_0, -WEIGHT_CLIP, WEIGHT_CLIP)
    model.bias_1 .= clamp.(model.bias_1, -BIAS_CLIP, BIAS_CLIP)
    model.bias_0 .= clamp.(model.bias_0, -BIAS_CLIP, BIAS_CLIP)
    sync_graph_readout!(model)

    return (
        loss = sum(abs2, y .- free_output) / 2,
        correct = argmax(class_scores(free_output)) == argmax(class_scores(y)),
        skipped = false,
    )
end

"""
    evaluate(model, x, y)

Evaluate the paper-like network with free-phase sampling only.
"""
function evaluate(model::M, x::X, y::Y) where {M<:PaperMLP,X<:AbstractMatrix,Y<:AbstractMatrix}
    correct = 0
    loss = 0f0
    pred_counts = zeros(Int, NCLASSES)
    confusion = zeros(Int, NCLASSES, NCLASSES)
    for sample_idx in axes(x, 2)
        free_state, _ = sample_phase!(model, view(x, :, sample_idx); reads = FREE_READS, sweeps = FREE_SWEEPS)
        output = @view free_state[model.output_idxs]
        target = view(y, :, sample_idx)
        pred = argmax(class_scores(output))
        truth = argmax(class_scores(target))
        pred_counts[pred] += 1
        confusion[truth, pred] += 1
        correct += pred == truth
        loss += sum(abs2, target .- output) / 2
    end
    per_class_accuracy = [confusion[digit, digit] / max(sum(view(confusion, digit, :)), 1) for digit in 1:NCLASSES]
    return (; accuracy = correct / size(x, 2), loss = loss / size(x, 2), pred_counts, confusion, per_class_accuracy)
end

"""
    matrix_csv(value)

Encode a small integer matrix as a CSV-safe row string using `;` between rows.
"""
function matrix_csv(value::T) where {T<:AbstractMatrix}
    rows = String[]
    sizehint!(rows, size(value, 1))
    for row in axes(value, 1)
        push!(rows, join(view(value, row, :), ":"))
    end
    return join(rows, ";")
end

"""
    save_model(path, model)

Serialize paper-like trainable parameters, including the external input matrix.
"""
function save_model(path::P, model::M) where {P<:AbstractString,M<:PaperMLP}
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (; weights_0 = model.weights_0, weights_1 = model.weights_1, bias_0 = model.bias_0, bias_1 = model.bias_1))
    end
    return path
end

"""
    load_model!(model, path)

Load a saved paper-like parameter snapshot into an initialized model.
"""
function load_model!(model::M, path::P) where {M<:PaperMLP,P<:AbstractString}
    params = open(deserialize, path)
    model.weights_0 .= params.weights_0
    model.weights_1 .= params.weights_1
    model.bias_0 .= params.bias_0
    model.bias_1 .= params.bias_1
    sync_graph_readout!(model)
    return model
end

"""
    main()

Train the paper-like Ising EP MLP and log epoch metrics.
"""
function main()
    xtrain, ytrain = EPOCHS > 0 ? balanced_mnist(:train, TRAIN_PER_CLASS) : (Matrix{Float32}(undef, INPUT_DIM, 0), Matrix{Float32}(undef, NCLASSES * OUTPUT_REPLICAS, 0))
    xtest, ytest = balanced_mnist(:test, TEST_PER_CLASS)
    model = init_model(1234)
    isempty(LOAD_PATH) || load_model!(model, LOAD_PATH)
    csv_path = joinpath(OUTDIR, "mnist_paper_like_ep.csv")
    best_accuracy = Ref(-Inf)
    best_path = joinpath(OUTDIR, "best_model.bin")
    final_path = joinpath(OUTDIR, "final_model.bin")

    open(joinpath(OUTDIR, "paper_like_settings.md"), "w") do io
        println(io, "# Paper-Like MNIST EP")
        println(io, "- hidden: `$(HIDDEN)`")
        println(io, "- output replicas: `$(OUTPUT_REPLICAS)`")
        println(io, "- train/test per class: `$(TRAIN_PER_CLASS)` / `$(TEST_PER_CLASS)`")
        println(io, "- free/nudge reads: `$(FREE_READS)` / `$(NUDGE_READS)`")
        println(io, "- free/nudge sweeps: `$(FREE_SWEEPS)` / `$(NUDGE_SWEEPS)`")
        println(io, "- beta: `$(BETA)`")
        println(io, "- learning rates W0/W1/B0/B1: `$(LR_W0)`, `$(LR_W1)`, `$(LR_B0)`, `$(LR_B1)`")
        println(io, "- gains W0/W1: `$(GAIN_W0)`, `$(GAIN_W1)`")
        println(io, "- clips weight/bias/applied_bias: `$(WEIGHT_CLIP)`, `$(BIAS_CLIP)`, `$(APPLIED_BIAS_CLIP)`")
        println(io, "- temps hot/reverse/cold: `$(HOT_TEMP)`, `$(REVERSE_TEMP)`, `$(COLD_TEMP)`")
        println(io, "- loaded checkpoint: `$(isempty(LOAD_PATH) ? "none" : LOAD_PATH)`")
    end

    for epoch in 0:EPOCHS
        seconds = 0.0
        train_accuracy = missing
        train_loss = missing
        skipped = missing
        if epoch > 0
            order = Random.shuffle(model.rng, collect(axes(xtrain, 2)))
            ncorrect = 0
            total_loss = 0f0
            nskipped = 0
            seconds = @elapsed begin
                for sample_idx in order
                    stats = train_one!(model, view(xtrain, :, sample_idx), view(ytrain, :, sample_idx))
                    ncorrect += stats.correct
                    total_loss += stats.loss
                    nskipped += stats.skipped
                end
            end
            train_accuracy = ncorrect / length(order)
            train_loss = total_loss / length(order)
            skipped = nskipped
        end
        test = evaluate(model, xtest, ytest)
        if test.accuracy > best_accuracy[]
            best_accuracy[] = test.accuracy
            save_model(best_path, model)
        end
        row = (;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            epoch,
            seconds,
            train_accuracy,
            train_loss,
            skipped,
            test_accuracy = test.accuracy,
            test_loss = test.loss,
            pred_counts = join(test.pred_counts, "-"),
            per_class_accuracy = join(round.(test.per_class_accuracy; digits = 3), "-"),
            confusion = matrix_csv(test.confusion),
            best_accuracy = best_accuracy[],
            best_path,
            final_path = epoch == EPOCHS ? final_path : "",
        )
        append_row!(csv_path, row)
        println(row)
        flush(stdout)
    end
    save_model(final_path, model)
    println("Saved paper-like MNIST EP run in ", OUTDIR)
end

main()
