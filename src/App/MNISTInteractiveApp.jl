module MNISTInteractiveApp

using GLMakie
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Windows
using MLDatasets
using Observables
using Printf
using Random
using Serialization

const II = IsingLearning.InteractiveIsing
const WIN = IsingLearning.InteractiveIsing.Windows
const PROC = IsingLearning.InteractiveIsing.Processes
const DEMO_FT = Float32

"""
    MNISTInteractiveConfig(; kwargs...)

Store the user-facing MNIST demo settings used to construct the graph,
simulation dynamics, data source, and GLMakie window.
"""
struct MNISTInteractiveConfig{FT<:AbstractFloat,P<:AbstractString,S<:Tuple{Int,Int}}
    hidden::Int
    output_replicas::Int
    temp::FT
    temp_slider_max::FT
    temp_slider_step::FT
    stepsize::FT
    weight_scale::FT
    checkpoint::P
    rng_seed::Int
    data_rng_seed::Int
    fps::Float64
    polling_rate::Float64
    size::S
end

"""
    MNISTInteractiveConfig(; kwargs...) -> MNISTInteractiveConfig

Construct a typed MNIST app configuration from keyword settings.
"""
function MNISTInteractiveConfig(;
    hidden::Integer,
    output_replicas::Integer,
    temp::FT,
    temp_slider_max::FT,
    temp_slider_step::FT,
    stepsize::FT,
    weight_scale::FT,
    checkpoint::P,
    rng_seed::Integer,
    data_rng_seed::Integer,
    fps::Real,
    polling_rate::Real,
    size::S,
) where {FT<:AbstractFloat,P<:AbstractString,S<:Tuple{Int,Int}}
    return MNISTInteractiveConfig(
        Int(hidden),
        Int(output_replicas),
        temp,
        temp_slider_max,
        temp_slider_step,
        stepsize,
        weight_scale,
        checkpoint,
        Int(rng_seed),
        Int(data_rng_seed),
        Float64(fps),
        Float64(polling_rate),
        size,
    )
end

"""
    MNISTInteractiveRuntime(...)

Hold the live state owned by one desktop MNIST demo window.
"""
mutable struct MNISTInteractiveRuntime{C,G,D,X,L,R,V,I,O,LO,DO,SO}
    config::C
    graph::G
    dynamics::D
    xdata::X
    labels::L
    digit_indices::V
    rng::R
    process_ref::Base.RefValue{Any}
    image_obs::I
    output_obs::O
    label_obs::LO
    selected_digit::DO
    status_obs::SO
end

"""
    default_checkpoint_path() -> String

Return the repository-local checkpoint used by the MNIST interactive demo.
"""
function default_checkpoint_path()
    return joinpath(
        @__DIR__,
        "..",
        "..",
        "ext",
        "IsingLearning",
        "ExperimentsOld",
        "mnist_manager",
        "runs",
        "20260520_paper_like_continue500pc_lrhalf_e8",
        "best_model.bin",
    )
end

"""
    default_config() -> MNISTInteractiveConfig

Build the demo configuration from environment variables and repository
defaults.
"""
function default_config()
    return MNISTInteractiveConfig(
        hidden = parse(Int, get(ENV, "ISING_MNIST_DEMO_HIDDEN", "120")),
        output_replicas = parse(Int, get(ENV, "ISING_MNIST_DEMO_OUTPUT_REPLICAS", "4")),
        temp = parse(DEMO_FT, get(ENV, "ISING_MNIST_DEMO_TEMP", "0.05")),
        temp_slider_max = parse(DEMO_FT, get(ENV, "ISING_MNIST_DEMO_TEMP_SLIDER_MAX", "1.0")),
        temp_slider_step = parse(DEMO_FT, get(ENV, "ISING_MNIST_DEMO_TEMP_SLIDER_STEP", "0.001")),
        stepsize = parse(DEMO_FT, get(ENV, "ISING_MNIST_DEMO_STEPSIZE", "0.001")),
        weight_scale = parse(DEMO_FT, get(ENV, "ISING_MNIST_DEMO_WEIGHT_SCALE", "0.01")),
        checkpoint = get(ENV, "ISING_MNIST_DEMO_CHECKPOINT", default_checkpoint_path()),
        rng_seed = parse(Int, get(ENV, "ISING_MNIST_DEMO_RNG_SEED", "42")),
        data_rng_seed = parse(Int, get(ENV, "ISING_MNIST_DEMO_DATA_RNG_SEED", "2026")),
        fps = parse(Float64, get(ENV, "ISING_MNIST_DEMO_FPS", "30")),
        polling_rate = parse(Float64, get(ENV, "ISING_MNIST_DEMO_POLLING_RATE", "10")),
        size = (1250, 900),
    )
end

"""
    normalize_mnist_images(images) -> Matrix{Float32}

Convert raw MNIST images to a feature-by-sample matrix in the demo precision.
"""
function normalize_mnist_images(images::T) where {T}
    x = DEMO_FT.(images)
    maximum(x) > one(DEMO_FT) && (x ./= DEMO_FT(255))
    return reshape(x, :, size(images, ndims(images)))
end

"""
    normalize_mnist_for_display(x) -> Matrix{Float32}

Rescale a flattened MNIST image to a stable `[0, 1]` display image.
"""
function normalize_mnist_for_display(x::T) where {T<:AbstractVector}
    image = reshape(x, D_MNIST, D_MNIST)
    lo = minimum(image)
    hi = maximum(image)
    hi == lo && return fill(DEMO_FT(0.5), size(image))
    return (image .- lo) ./ (hi - lo)
end

"""
    mnist_class_scores(output, replicas) -> Vector{Float32}

Average replicated output units into one score per MNIST digit.
"""
function mnist_class_scores(output::T, replicas::Int) where {T<:AbstractVector}
    scores = zeros(DEMO_FT, MNIST_NCLASSES)
    @inbounds for digit in 1:MNIST_NCLASSES
        first_idx = (digit - 1) * replicas + 1
        scores[digit] = sum(view(output, first_idx:(first_idx + replicas - 1))) / replicas
    end
    return scores
end

"""
    mnist_status(runtime, running) -> String

Format the label, prediction, output strength, temperature, and run state for
the status label.
"""
function mnist_status(runtime::T, running::Bool) where {T<:MNISTInteractiveRuntime}
    output = mnist_class_scores(vec(II.state(runtime.graph[end])), runtime.config.output_replicas)
    prediction = argmax(output) - 1
    maxval = isempty(output) ? zero(DEMO_FT) : maximum(output)
    return @sprintf(
        "label = %d    prediction = %d    max output = % .3f    T = %.4g    %s",
        runtime.label_obs[],
        prediction,
        maxval,
        II.temp(runtime.graph),
        running ? "running" : "idle",
    )
end

"""
    install_paper_mnist_parameters!(graph, path) -> Any

Load a paper-style MNIST checkpoint into an interactive graph, installing the
input-hidden and hidden-output couplings plus hidden/output biases.
"""
function install_paper_mnist_parameters!(graph::G, path::P) where {G,P<:AbstractString}
    isfile(path) || throw(ArgumentError("MNIST checkpoint does not exist: $path"))
    params = open(deserialize, path)

    input_idxs = collect(II.layerrange(graph[1]))
    hidden_idxs = collect(II.layerrange(graph[2]))
    output_idxs = collect(II.layerrange(graph[3]))

    size(params.weights_0) == (length(input_idxs), length(hidden_idxs)) ||
        throw(ArgumentError("weights_0 has size $(size(params.weights_0)), expected $((length(input_idxs), length(hidden_idxs)))"))
    size(params.weights_1) == (length(hidden_idxs), length(output_idxs)) ||
        throw(ArgumentError("weights_1 has size $(size(params.weights_1)), expected $((length(hidden_idxs), length(output_idxs)))"))
    length(params.bias_0) == length(hidden_idxs) ||
        throw(ArgumentError("bias_0 length $(length(params.bias_0)) does not match hidden layer length $(length(hidden_idxs))"))
    length(params.bias_1) == length(output_idxs) ||
        throw(ArgumentError("bias_1 length $(length(params.bias_1)) does not match output layer length $(length(output_idxs))"))

    # Install trained couplings in the sign convention expected by the Ising Hamiltonian.
    A = II.adj(graph)
    @inbounds for hpos in eachindex(hidden_idxs)
        hidden_idx = hidden_idxs[hpos]
        for ipos in eachindex(input_idxs)
            A[hidden_idx, input_idxs[ipos]] = -DEMO_FT(params.weights_0[ipos, hpos])
        end
    end
    @inbounds for opos in eachindex(output_idxs)
        output_idx = output_idxs[opos]
        for hpos in eachindex(hidden_idxs)
            A[output_idx, hidden_idxs[hpos]] = -DEMO_FT(params.weights_1[hpos, opos])
        end
    end

    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    b[input_idxs] .= zero(DEMO_FT)
    b[hidden_idxs] .= .-DEMO_FT.(params.bias_0)
    b[output_idxs] .= .-DEMO_FT.(params.bias_1)
    return params
end

"""
    build_mnist_graph(config) -> AbstractIsingGraph

Construct the MNIST graph, assign display coordinates, set temperature, and
load trained checkpoint parameters.
"""
function build_mnist_graph(config::T) where {T<:MNISTInteractiveConfig}
    graph = MNISTArchitecture(
        hidden = config.hidden,
        output_replicas = config.output_replicas,
        precision = DEMO_FT,
        weight_scale = config.weight_scale,
        rng = Random.MersenneTwister(config.rng_seed),
    )
    II.setcoords!(graph[1]; x = 0, y = 0, z = 0)
    II.setcoords!(graph[2]; x = 90, y = 0, z = 0)
    II.setcoords!(graph[3]; x = 530, y = 0, z = 0)
    II.temp!(graph, config.temp)
    II.off!(graph.index_set, 1)
    install_paper_mnist_parameters!(graph, config.checkpoint)
    return graph
end

"""
    load_mnist_data() -> Tuple{Matrix{Float32}, Vector{Int}, Vector{Vector{Int}}}

Load MNIST training images, normalize them for graph input, and index examples
by digit.
"""
function load_mnist_data()
    images, raw_labels = MLDatasets.MNIST(split = :train)[:]
    xdata = Matrix(normalize_mnist_images(images))
    labels = Int.(raw_labels)
    digit_indices = [findall(==(digit), labels) for digit in 0:9]
    return xdata, labels, digit_indices
end

"""
    build_runtime(config = default_config()) -> MNISTInteractiveRuntime

Create the graph, dynamics, data arrays, random state, and display observables
used by one MNIST demo window.
"""
function build_runtime(config::T = default_config()) where {T<:MNISTInteractiveConfig}
    graph = build_mnist_graph(config)
    dynamics = LocalLangevin(
        stepsize = config.stepsize,
        adjusted = false,
        group_steps = 1,
    )
    xdata, labels, digit_indices = load_mnist_data()

    runtime = MNISTInteractiveRuntime(
        config,
        graph,
        dynamics,
        xdata,
        labels,
        digit_indices,
        Random.MersenneTwister(config.data_rng_seed),
        Ref{Any}(nothing),
        Observable(zeros(DEMO_FT, D_MNIST, D_MNIST)),
        Observable(zeros(DEMO_FT, 10)),
        Observable(-1),
        Observable(0),
        Observable(""),
    )
    runtime.status_obs[] = mnist_status(runtime, false)
    return runtime
end

"""
    process_running(runtime) -> Bool

Return whether any simulation process attached to the runtime graph is active.
"""
function process_running(runtime::T) where {T<:MNISTInteractiveRuntime}
    return any(PROC.isrunning, II.processes(runtime.graph))
end

"""
    stop_graph_processes!(runtime) -> Nothing

Request all graph simulation processes to stop and remove them from the graph's
process list.
"""
function stop_graph_processes!(runtime::T) where {T<:MNISTInteractiveRuntime}
    WIN._request_graph_process_close!(runtime.graph)
    runtime.process_ref[] = nothing
    return nothing
end

"""
    dispatch_mnist_index!(runtime, idx) -> Nothing

Load one MNIST sample into the input layer, start a fresh Langevin process, and
refresh the display observables.
"""
function dispatch_mnist_index!(runtime::T, idx::Int) where {T<:MNISTInteractiveRuntime}
    stop_graph_processes!(runtime)

    # Copy the selected input so the graph state is isolated from the shared data matrix.
    x = copy(@view runtime.xdata[:, idx])
    label = runtime.labels[idx]
    II.state(runtime.graph[1]) .= reshape(x, size(II.state(runtime.graph[1])))
    II.off!(runtime.graph.index_set, 1)

    runtime.image_obs[] = normalize_mnist_for_display(x)
    runtime.label_obs[] = label
    graph_dynamics = deepcopy(runtime.dynamics)
    process = PROC.Process(
        graph_dynamics,
        PROC.Init(graph_dynamics, model = runtime.graph),
    )
    push!(II.processes(runtime.graph), process)
    PROC.run(process)
    runtime.process_ref[] = process
    runtime.status_obs[] = mnist_status(runtime, true)
    return nothing
end

"""
    dispatch_selected_digit!(runtime) -> Nothing

Load a random MNIST sample matching the currently selected digit.
"""
function dispatch_selected_digit!(runtime::T) where {T<:MNISTInteractiveRuntime}
    digit = runtime.selected_digit[]
    idxs = runtime.digit_indices[digit + 1]
    isempty(idxs) && return nothing
    return dispatch_mnist_index!(runtime, rand(runtime.rng, idxs))
end

"""
    dispatch_random_mnist!(runtime) -> Nothing

Load a random MNIST sample from the configured data matrix.
"""
function dispatch_random_mnist!(runtime::T) where {T<:MNISTInteractiveRuntime}
    return dispatch_mnist_index!(runtime, rand(runtime.rng, axes(runtime.xdata, 2)))
end

"""
    build_window(runtime) -> WindowHost

Create the GLMakie window, connect controls to the runtime, and start the first
random MNIST simulation.
"""
function build_window(runtime::T) where {T<:MNISTInteractiveRuntime}
    host = window(
        title = "Interactive MNIST Debug",
        size = runtime.config.size,
        fps = runtime.config.fps,
        polling_rate = runtime.config.polling_rate,
    )

    # Top controls select data and tune the live simulation temperature.
    controls = GridLayout(host.figure[1, 1])
    digit_slider = Slider(controls[1, 1], range = 0:9, startvalue = runtime.selected_digit[], width = 220)
    Label(controls[1, 2], lift(digit -> "Digit $digit", runtime.selected_digit); width = 70)
    digit_button = Button(controls[1, 3], label = "Load Digit", width = 120, height = 34)
    random_button = Button(controls[1, 4], label = "Random Any", width = 120, height = 34)
    Label(controls[1, 5], runtime.status_obs; tellwidth = false, halign = :left)
    Label(controls[2, 1], "T"; tellwidth = false, halign = :right)
    temperature_slider = Slider(
        controls[2, 2:4],
        range = 0:runtime.config.temp_slider_step:runtime.config.temp_slider_max,
        startvalue = runtime.config.temp,
        width = 360,
    )
    Label(controls[2, 5], lift(value -> @sprintf("%.4g", value), temperature_slider.value); tellwidth = false, halign = :left)
    ups, upsps = WIN._steps_per_second_observables!(host, runtime.graph)
    Label(controls[3, 1:2], "Steps per second"; tellwidth = false, halign = :left)
    Label(controls[3, 3], lift(value -> "$(round(value, digits = 2))", ups); tellwidth = false, halign = :left)
    Label(controls[3, 4], "per unit"; tellwidth = false, halign = :right)
    Label(controls[3, 5], lift(value -> "$(round(value, digits = 2))", upsps); tellwidth = false, halign = :left)

    # Bottom plots show the selected input and the current output-layer scores.
    image_axis = Axis(host.figure[3, 1], title = "input image", aspect = DataAspect())
    image_axis.yreversed = true
    hidedecorations!(image_axis)
    heatmap!(image_axis, runtime.image_obs; colormap = :grays)

    output_axis = Axis(host.figure[3, 2], title = "output layer", xlabel = "digit", ylabel = "state", xticks = 0:9)
    barplot!(output_axis, 0:9, runtime.output_obs; color = :dodgerblue)

    panel!(
        host,
        AllLayersViewPanel(
            runtime.graph;
            colormap = :thermal,
            labels = true,
            display_sizes = ((70, 70), (420, 420), (140, 90)),
            axis_kwargs = (
                title = "MNIST 28^2 -> $(runtime.config.hidden) -> $(10 * runtime.config.output_replicas)",
                aspect = nothing,
            ),
        ),
        (2, 1:2),
    )

    # Callbacks keep the controls and Makie observables synchronized with the graph.
    register!(host, on(digit_slider.value) do value
        runtime.selected_digit[] = Int(round(value))
        return nothing
    end)
    register!(host, on(digit_button.clicks) do _
        dispatch_selected_digit!(runtime)
        return nothing
    end)
    register!(host, on(random_button.clicks) do _
        dispatch_random_mnist!(runtime)
        return nothing
    end)
    register!(host, on(temperature_slider.value) do value
        WIN._set_temperature!(runtime.graph, DEMO_FT(value))
        runtime.status_obs[] = mnist_status(runtime, process_running(runtime))
        return nothing
    end)
    register_frame!(host) do _
        runtime.output_obs[] = mnist_class_scores(vec(II.state(runtime.graph[end])), runtime.config.output_replicas)
        runtime.status_obs[] = mnist_status(runtime, process_running(runtime))
        return nothing
    end
    onclose!(host) do _
        stop_graph_processes!(runtime)
        return nothing
    end

    dispatch_random_mnist!(runtime)
    return host
end

"""
    wait_until_closed(host) -> Nothing

Block the current task until the GLMakie host reports that its window has
closed.
"""
function wait_until_closed(host::T) where {T<:WIN.WindowHost}
    while !host.closed && host.open[]
        sleep(0.1)
    end
    return nothing
end

"""
    run_mnist_interactive_app(; config = default_config(), wait_for_close = true)

Start the GLMakie MNIST interactive app and optionally block until the window is
closed.
"""
function run_mnist_interactive_app(; config::T = default_config(), wait_for_close::Bool = true) where {T<:MNISTInteractiveConfig}
    Threads.nthreads() == 4 || @info "MNIST app is configured for 4 Julia threads; restart with `julia -t 4` for the intended desktop build." threads = Threads.nthreads()
    GLMakie.activate!()
    runtime = build_runtime(config)
    host = build_window(runtime)
    wait_for_close && wait_until_closed(host)
    return (; host, runtime)
end

"""
    smoke_check(; config = default_config()) -> NamedTuple

Build the runtime, launch one short background simulation, stop it, and return
the observed run states without opening a GLMakie window.
"""
function smoke_check(; config::T = default_config()) where {T<:MNISTInteractiveConfig}
    runtime = build_runtime(config)
    dispatch_random_mnist!(runtime)
    sleep(0.2)
    running_before_stop = process_running(runtime)
    process_count_before_stop = length(II.processes(runtime.graph))
    label = runtime.label_obs[]
    stop_graph_processes!(runtime)
    sleep(0.2)
    running_after_stop = process_running(runtime)
    process_count_after_stop = length(II.processes(runtime.graph))
    return (;
        running_before_stop,
        process_count_before_stop,
        label,
        running_after_stop,
        process_count_after_stop,
    )
end

"""
    julia_main() -> Cint

PackageCompiler-compatible entrypoint for the MNIST desktop app.
"""
function julia_main()::Cint
    try
        run_mnist_interactive_app()
        return 0
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        return 1
    end
end

export MNISTInteractiveConfig, build_runtime, build_window, default_config,
    julia_main, run_mnist_interactive_app, smoke_check

end
