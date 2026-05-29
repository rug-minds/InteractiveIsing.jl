using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "ext", "IsingLearning"))

using GLMakie
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Windows
using MLDatasets
using Printf
using Random
using Serialization

GLMakie.activate!()

const LOCAL_MNIST_WINDOWS = IsingLearning.InteractiveIsing.Windows
const LOCAL_MNIST_PROCESSES = IsingLearning.InteractiveIsing.Processes

const LOCAL_MNIST_FT = Float32
const LOCAL_MNIST_HIDDEN1_SIDE = parse(Int, get(ENV, "ISING_MNIST_LOCAL_DEMO_H1_SIDE", "28"))
const LOCAL_MNIST_HIDDEN2_SIDE = parse(Int, get(ENV, "ISING_MNIST_LOCAL_DEMO_H2_SIDE", "11"))
const LOCAL_MNIST_OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_LOCAL_DEMO_OUTPUT_REPLICAS", "4"))
const LOCAL_MNIST_TEMP = parse(LOCAL_MNIST_FT, get(ENV, "ISING_MNIST_LOCAL_DEMO_TEMP", "0.01"))
const LOCAL_MNIST_TEMP_SLIDER_MAX = parse(LOCAL_MNIST_FT, get(ENV, "ISING_MNIST_LOCAL_DEMO_TEMP_SLIDER_MAX", "1.0"))
const LOCAL_MNIST_TEMP_SLIDER_STEP = parse(LOCAL_MNIST_FT, get(ENV, "ISING_MNIST_LOCAL_DEMO_TEMP_SLIDER_STEP", "0.001"))
const LOCAL_MNIST_STEPSIZE = parse(LOCAL_MNIST_FT, get(ENV, "ISING_MNIST_LOCAL_DEMO_STEPSIZE", "0.001"))
const LOCAL_MNIST_CHECKPOINT = get(
    ENV,
    "ISING_MNIST_LOCAL_DEMO_CHECKPOINT",
    joinpath(
        @__DIR__,
        "..",
        "..",
        "ext",
        "IsingLearning",
        "ExperimentsOld",
        "mnist_manager",
        "runs",
        "20260522_local_paper_h28_h11_1000pc_continue_lr0001_reads5",
        "best_model.bin",
    ),
)

"""Normalize raw MNIST images into a feature-by-sample matrix."""
function normalize_local_mnist_images(images::T) where {T}
    x = LOCAL_MNIST_FT.(images)
    maximum(x) > one(LOCAL_MNIST_FT) && (x ./= LOCAL_MNIST_FT(255))
    return reshape(x, :, size(images, ndims(images)))
end

"""Normalize one flattened MNIST image for the preview plot."""
function normalize_local_mnist_display(x::T) where {T<:AbstractVector}
    image = reshape(x, D_MNIST, D_MNIST)
    lo = minimum(image)
    hi = maximum(image)
    hi == lo && return fill(LOCAL_MNIST_FT(0.5), size(image))
    return (image .- lo) ./ (hi - lo)
end

"""Return a compact 2D layer shape for a flat number of units."""
function local_mnist_factor_shape(units::Integer)
    rows = floor(Int, sqrt(Int(units)))
    while rows > 1 && Int(units) % rows != 0
        rows -= 1
    end
    return rows, Int(units) ÷ rows
end

"""Average replicated output units into one scalar score per digit."""
function local_mnist_class_scores(output::T) where {T<:AbstractVector}
    scores = zeros(LOCAL_MNIST_FT, MNIST_NCLASSES)
    @inbounds for digit in 1:MNIST_NCLASSES
        first_idx = (digit - 1) * LOCAL_MNIST_OUTPUT_REPLICAS + 1
        scores[digit] = sum(view(output, first_idx:(first_idx + LOCAL_MNIST_OUTPUT_REPLICAS - 1))) / LOCAL_MNIST_OUTPUT_REPLICAS
    end
    return scores
end

"""Format the live classifier state for the top status line."""
function local_mnist_status(graph::G, label::Integer, running::Bool) where {G}
    output = local_mnist_class_scores(vec(InteractiveIsing.state(graph[end])))
    prediction = argmax(output) - 1
    maxval = isempty(output) ? zero(LOCAL_MNIST_FT) : maximum(output)
    return @sprintf(
        "label = %d    prediction = %d    max output = % .3f    T = %.4g    %s",
        label,
        prediction,
        maxval,
        InteractiveIsing.temp(graph),
        running ? "running" : "idle",
    )
end

"""Construct the local paper-like graph used by the saved MNIST checkpoint."""
function build_local_mnist_demo_graph()
    output_rows, output_cols = local_mnist_factor_shape(MNIST_NCLASSES * LOCAL_MNIST_OUTPUT_REPLICAS)
    zero_weights = AllToAllWeightGenerator((; dr, c1, c2, dc) -> zero(LOCAL_MNIST_FT))
    input = InteractiveIsing.Layer(
        D_MNIST,
        D_MNIST,
        InteractiveIsing.StateSet(-1f0, 1f0),
        InteractiveIsing.Continuous(),
        InteractiveIsing.Coords(0, 0, 0);
        periodic = false,
    )
    hidden1 = InteractiveIsing.Layer(
        LOCAL_MNIST_HIDDEN1_SIDE,
        LOCAL_MNIST_HIDDEN1_SIDE,
        InteractiveIsing.StateSet(-1f0, 1f0),
        InteractiveIsing.Continuous(),
        InteractiveIsing.Coords(0, 90, 0);
        periodic = false,
    )
    hidden2 = InteractiveIsing.Layer(
        LOCAL_MNIST_HIDDEN2_SIDE,
        LOCAL_MNIST_HIDDEN2_SIDE,
        InteractiveIsing.StateSet(-1f0, 1f0),
        InteractiveIsing.Continuous(),
        InteractiveIsing.Coords(0, 540, 0);
        periodic = false,
    )
    output = InteractiveIsing.Layer(
        output_rows,
        output_cols,
        InteractiveIsing.StateSet(-1f0, 1f0),
        InteractiveIsing.Continuous(),
        InteractiveIsing.Coords(0, 710, 0);
        periodic = false,
    )
    graph = InteractiveIsing.IsingGraph(
        input,
        zero_weights,
        hidden1,
        deepcopy(zero_weights),
        hidden2,
        deepcopy(zero_weights),
        output,
        InteractiveIsing.Bilinear() + InteractiveIsing.MagField(b = g -> InteractiveIsing.filltype(Vector, 0f0, InteractiveIsing.statelen(g)));
        index_set = g -> InteractiveIsing.ToggledIndexSet(g),
    )
    InteractiveIsing.temp!(graph, LOCAL_MNIST_TEMP)
    InteractiveIsing.off!(graph.index_set, 1)
    return graph
end

"""
    local_mnist_checkpoint_params(path)

Load either a direct parameter tuple or a saved experiment tuple containing a
`params` field.
"""
function local_mnist_checkpoint_params(path::P) where {P<:AbstractString}
    isfile(path) || throw(ArgumentError("Local MNIST checkpoint does not exist: $path"))
    raw = open(deserialize, path)
    return hasproperty(raw, :params) ? raw.params : raw
end

"""Install the saved local MNIST weights and biases into `graph`."""
function install_local_mnist_parameters!(graph::G, path::P) where {G,P<:AbstractString}
    params = local_mnist_checkpoint_params(path)
    input_idxs = collect(InteractiveIsing.layerrange(graph[1]))
    h1_idxs = collect(InteractiveIsing.layerrange(graph[2]))
    h2_idxs = collect(InteractiveIsing.layerrange(graph[3]))
    output_idxs = collect(InteractiveIsing.layerrange(graph[4]))

    size(params.weights_0) == (length(input_idxs), length(h1_idxs)) ||
        throw(ArgumentError("weights_0 has size $(size(params.weights_0)), expected $((length(input_idxs), length(h1_idxs)))"))
    size(params.weights_12) == (length(h1_idxs), length(h2_idxs)) ||
        throw(ArgumentError("weights_12 has size $(size(params.weights_12)), expected $((length(h1_idxs), length(h2_idxs)))"))
    size(params.weights_2o) == (length(h2_idxs), length(output_idxs)) ||
        throw(ArgumentError("weights_2o has size $(size(params.weights_2o)), expected $((length(h2_idxs), length(output_idxs)))"))

    A = InteractiveIsing.adj(graph)

    # Checkpoints store the learning-sign convention; graph couplings use the
    # Hamiltonian convention, so installed couplings are negated.
    @inbounds for h1pos in eachindex(h1_idxs), ipos in eachindex(input_idxs)
        w = -LOCAL_MNIST_FT(params.weights_0[ipos, h1pos])
        w == 0f0 && continue
        A[h1_idxs[h1pos], input_idxs[ipos]] = w
        A[input_idxs[ipos], h1_idxs[h1pos]] = w
    end
    @inbounds for h2pos in eachindex(h2_idxs), h1pos in eachindex(h1_idxs)
        w = -LOCAL_MNIST_FT(params.weights_12[h1pos, h2pos])
        w == 0f0 && continue
        A[h2_idxs[h2pos], h1_idxs[h1pos]] = w
        A[h1_idxs[h1pos], h2_idxs[h2pos]] = w
    end
    @inbounds for opos in eachindex(output_idxs), h2pos in eachindex(h2_idxs)
        w = -LOCAL_MNIST_FT(params.weights_2o[h2pos, opos])
        w == 0f0 && continue
        A[output_idxs[opos], h2_idxs[h2pos]] = w
        A[h2_idxs[h2pos], output_idxs[opos]] = w
    end

    @inbounds for dstpos in eachindex(h1_idxs), srcpos in eachindex(h1_idxs)
        dstpos == srcpos && continue
        w = -LOCAL_MNIST_FT(params.weights_11[srcpos, dstpos])
        w == 0f0 && continue
        A[h1_idxs[dstpos], h1_idxs[srcpos]] = w
    end
    @inbounds for dstpos in eachindex(h2_idxs), srcpos in eachindex(h2_idxs)
        dstpos == srcpos && continue
        w = -LOCAL_MNIST_FT(params.weights_22[srcpos, dstpos])
        w == 0f0 && continue
        A[h2_idxs[dstpos], h2_idxs[srcpos]] = w
    end
    @inbounds for dstpos in eachindex(output_idxs), srcpos in eachindex(output_idxs)
        dstpos == srcpos && continue
        w = -LOCAL_MNIST_FT(params.weights_oo[srcpos, dstpos])
        w == 0f0 && continue
        A[output_idxs[dstpos], output_idxs[srcpos]] = w
    end

    b = InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
    b[input_idxs] .= zero(LOCAL_MNIST_FT)
    b[h1_idxs] .= .-LOCAL_MNIST_FT.(params.bias_1)
    b[h2_idxs] .= .-LOCAL_MNIST_FT.(params.bias_2)
    b[output_idxs] .= .-LOCAL_MNIST_FT.(params.bias_o)
    return params
end

"""Reset only the simulated layers before loading a new MNIST image."""
function randomize_local_mnist_state!(graph::G, rng::R) where {G,R<:Random.AbstractRNG}
    for layer_idx in 2:length(InteractiveIsing.layers(graph))
        s = InteractiveIsing.state(graph[layer_idx])
        @inbounds for idx in eachindex(s)
            s[idx] = rand(rng, Bool) ? 1f0 : -1f0
        end
    end
    return graph
end

graph = build_local_mnist_demo_graph()
checkpoint_params = Base.invokelatest(install_local_mnist_parameters!, graph, LOCAL_MNIST_CHECKPOINT)

dynamics = LocalLangevin(
    stepsize = LOCAL_MNIST_STEPSIZE,
    adjusted = false,
    group_steps = 1,
    order = :cyclic,
)
images, raw_labels = MLDatasets.MNIST(split = :train)[:]
xdata = Matrix(normalize_local_mnist_images(images))
rng = Random.MersenneTwister(2026)
labels = Int.(raw_labels)
digit_indices = [findall(==(digit), labels) for digit in 0:9]
process_ref = Ref{Any}(nothing)

image_obs = Observable(zeros(LOCAL_MNIST_FT, D_MNIST, D_MNIST))
output_obs = Observable(zeros(LOCAL_MNIST_FT, MNIST_NCLASSES))
label_obs = Observable(-1)
selected_digit = Observable(0)
status_obs = Observable(local_mnist_status(graph, label_obs[], false))

local_mnist_digit_label(digit) = "Digit $digit"
local_mnist_temperature_label(value) = @sprintf("%.4g", value)
local_mnist_rate_label(value) = "$(round(value, digits = 2))"

"""Return whether the demo graph currently has a running process."""
function local_mnist_process_running()
    return any(LOCAL_MNIST_PROCESSES.isrunning, InteractiveIsing.processes(graph))
end

"""Load one MNIST sample into the input layer and start continuous Langevin."""
function dispatch_local_mnist_index!(idx::Integer)
    LOCAL_MNIST_WINDOWS._request_graph_process_close!(graph)

    x = copy(@view xdata[:, Int(idx)])
    label = labels[Int(idx)]

    InteractiveIsing.state(graph[1]) .= reshape(x, size(InteractiveIsing.state(graph[1])))
    InteractiveIsing.off!(graph.index_set, 1)
    randomize_local_mnist_state!(graph, rng)

    image_obs[] = normalize_local_mnist_display(x)
    label_obs[] = label
    status_obs[] = local_mnist_status(graph, label, true)

    graph_dynamics = deepcopy(dynamics)
    process = LOCAL_MNIST_PROCESSES.Process(
        graph_dynamics,
        LOCAL_MNIST_PROCESSES.Init(graph_dynamics, model = graph),
    )
    push!(InteractiveIsing.processes(graph), process)
    LOCAL_MNIST_PROCESSES.run(process)
    process_ref[] = process
    return nothing
end

"""Load a random training sample with the currently selected digit."""
function dispatch_selected_local_mnist!()
    digit = selected_digit[]
    idxs = digit_indices[digit + 1]
    isempty(idxs) && return nothing
    return dispatch_local_mnist_index!(rand(rng, idxs))
end

"""Load a random MNIST training sample from any digit."""
function dispatch_random_local_mnist!()
    return dispatch_local_mnist_index!(rand(rng, axes(xdata, 2)))
end

"""Update the selected digit from the digit slider."""
function local_mnist_digit_slider_changed(value)
    selected_digit[] = Int(round(value))
    return nothing
end

"""Handle the selected-digit load button."""
function local_mnist_digit_button_clicked(_)
    dispatch_selected_local_mnist!()
    return nothing
end

"""Handle the random-image load button."""
function local_mnist_random_button_clicked(_)
    dispatch_random_local_mnist!()
    return nothing
end

"""Apply a new graph temperature from the UI slider."""
function local_mnist_temperature_changed(value)
    LOCAL_MNIST_WINDOWS._set_temperature!(graph, LOCAL_MNIST_FT(value))
    status_obs[] = local_mnist_status(graph, label_obs[], local_mnist_process_running())
    return nothing
end

"""Refresh live output bars and status text once per rendered frame."""
function local_mnist_frame!(_)
    output_obs[] = local_mnist_class_scores(vec(InteractiveIsing.state(graph[end])))
    status_obs[] = local_mnist_status(graph, label_obs[], local_mnist_process_running())
    return nothing
end

"""Close any running simulation when the window closes."""
function local_mnist_close!(_)
    LOCAL_MNIST_WINDOWS._request_graph_process_close!(graph)
    return nothing
end

host = window(
    title = "Interactive Local MNIST Debug",
    size = (1250, 900),
    fps = 30,
    polling_rate = 10,
)
LOCAL_MNIST_WINDOWS.register_hot_observable!(host, image_obs)
LOCAL_MNIST_WINDOWS.register_hot_observable!(host, output_obs)

controls = GridLayout(host.figure[1, 1])
digit_slider = Slider(controls[1, 1], range = 0:9, startvalue = selected_digit[], width = 220)
digit_label = Label(controls[1, 2], lift(local_mnist_digit_label, selected_digit); width = 70)
digit_button = Button(controls[1, 3], label = "Load Digit", width = 120, height = 34)
random_button = Button(controls[1, 4], label = "Random Any", width = 120, height = 34)
Label(controls[1, 5], status_obs; tellwidth = false, halign = :left)
Label(controls[2, 1], "T"; tellwidth = false, halign = :right)
temperature_slider = Slider(
    controls[2, 2:4],
    range = 0:LOCAL_MNIST_TEMP_SLIDER_STEP:LOCAL_MNIST_TEMP_SLIDER_MAX,
    startvalue = LOCAL_MNIST_TEMP,
    width = 360,
)
Label(controls[2, 5], lift(local_mnist_temperature_label, temperature_slider.value); tellwidth = false, halign = :left)
ups, upsps = LOCAL_MNIST_WINDOWS._steps_per_second_observables!(host, graph)
Label(controls[3, 1:2], "Steps per second"; tellwidth = false, halign = :left)
Label(controls[3, 3], lift(local_mnist_rate_label, ups); tellwidth = false, halign = :left)
Label(controls[3, 4], "per unit"; tellwidth = false, halign = :right)
Label(controls[3, 5], lift(local_mnist_rate_label, upsps); tellwidth = false, halign = :left)

layer_panel = panel!(
    host,
    AllLayersViewPanel(
        graph;
        colormap = :thermal,
        labels = true,
        display_sizes = (
            (70, 70),
            (420, 420),
            (150, 150),
            (120, 90),
        ),
        axis_kwargs = (
            title = "Local MNIST 28^2 -> $(LOCAL_MNIST_HIDDEN1_SIDE^2) -> $(LOCAL_MNIST_HIDDEN2_SIDE^2) -> $(10 * LOCAL_MNIST_OUTPUT_REPLICAS)",
            aspect = nothing,
        ),
    ),
    (2, 1:2),
)

image_axis = Axis(
    host.figure[3, 1],
    title = "input image",
    aspect = DataAspect(),
)
image_axis.yreversed = true
hidedecorations!(image_axis)
heatmap!(image_axis, image_obs; colormap = :grays)

output_axis = Axis(
    host.figure[3, 2],
    title = "output layer",
    xlabel = "digit",
    ylabel = "state",
    xticks = 0:9,
)
barplot!(output_axis, 0:9, output_obs; color = :dodgerblue)

register!(host, on(local_mnist_digit_slider_changed, digit_slider.value))
register!(host, on(local_mnist_digit_button_clicked, digit_button.clicks))
register!(host, on(local_mnist_random_button_clicked, random_button.clicks))
register!(host, on(local_mnist_temperature_changed, temperature_slider.value))
register_frame!(host, local_mnist_frame!)
onclose!(host, local_mnist_close!)

dispatch_random_local_mnist!()

local_mnist_demo = (;
    graph,
    host,
    layer_panel,
    process = process_ref,
    checkpoint_params,
)
