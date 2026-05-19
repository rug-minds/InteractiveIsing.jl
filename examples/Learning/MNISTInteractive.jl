using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "ext", "IsingLearning"))

using GLMakie
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Windows
using MLDatasets
using Printf
using Random

GLMakie.activate!()

const MNIST_DEMO_WINDOWS = IsingLearning.InteractiveIsing.Windows
const MNIST_DEMO_PROCESSES = IsingLearning.InteractiveIsing.Processes

const MNIST_DEMO_FT = Float32
const MNIST_DEMO_HIDDEN = parse(Int, get(ENV, "ISING_MNIST_DEMO_HIDDEN", string(7840)))
const MNIST_DEMO_TEMP = parse(MNIST_DEMO_FT, get(ENV, "ISING_MNIST_DEMO_TEMP", "0.05"))
const MNIST_DEMO_TEMP_SLIDER_MAX = parse(MNIST_DEMO_FT, get(ENV, "ISING_MNIST_DEMO_TEMP_SLIDER_MAX", "1.0"))
const MNIST_DEMO_TEMP_SLIDER_STEP = parse(MNIST_DEMO_FT, get(ENV, "ISING_MNIST_DEMO_TEMP_SLIDER_STEP", "0.001"))
const MNIST_DEMO_STEPSIZE = parse(MNIST_DEMO_FT, get(ENV, "ISING_MNIST_DEMO_STEPSIZE", "0.001"))
const MNIST_DEMO_WEIGHT_SCALE = parse(MNIST_DEMO_FT, get(ENV, "ISING_MNIST_DEMO_WEIGHT_SCALE", "0.01"))

function normalize_mnist_images(images, graph)
    states = InteractiveIsing.stateset(graph[1])
    lo = MNIST_DEMO_FT(first(states))
    hi = MNIST_DEMO_FT(last(states))
    x = MNIST_DEMO_FT.(images)
    maximum(x) > one(MNIST_DEMO_FT) && (x ./= MNIST_DEMO_FT(255))
    x .*= hi - lo
    x .+= lo
    return reshape(x, :, size(images, ndims(images)))
end

function normalize_mnist_for_display(x)
    image = reshape(x, D_MNIST, D_MNIST)
    lo = minimum(image)
    hi = maximum(image)
    hi == lo && return fill(MNIST_DEMO_FT(0.5), size(image))
    return (image .- lo) ./ (hi - lo)
end

function mnist_status(graph, label, running)
    output = collect(vec(InteractiveIsing.state(graph[end])))
    prediction = argmax(output) - 1
    maxval = isempty(output) ? zero(MNIST_DEMO_FT) : maximum(output)
    return @sprintf(
        "label = %d    prediction = %d    max output = % .3f    T = %.4g    %s",
        label,
        prediction,
        maxval,
        InteractiveIsing.temp(graph),
        running ? "running" : "idle",
    )
end

graph = MNISTArchitecture(
    hidden = MNIST_DEMO_HIDDEN,
    precision = MNIST_DEMO_FT,
    weight_scale = MNIST_DEMO_WEIGHT_SCALE,
    rng = Random.MersenneTwister(42),
)
InteractiveIsing.setcoords!(graph[1]; x = 0, y = 0, z = 0)
InteractiveIsing.setcoords!(graph[2]; x = 90, y = 0, z = 0)
InteractiveIsing.setcoords!(graph[3]; x = 530, y = 0, z = 0)
InteractiveIsing.temp!(graph, MNIST_DEMO_TEMP)
InteractiveIsing.off!(graph.index_set, 1)

dynamics = LocalLangevin(
    stepsize = MNIST_DEMO_STEPSIZE,
    adjusted = false,
    group_steps = 1,
)
images, raw_labels = MLDatasets.MNIST(split = :train)[:]
xdata = normalize_mnist_images(images, graph)
xdata = Matrix(xdata)
rng = Random.MersenneTwister(2026)
labels = Int.(raw_labels)
digit_indices = [findall(==(digit), labels) for digit in 0:9]
process_ref = Ref{Any}(nothing)

image_obs = Observable(zeros(MNIST_DEMO_FT, D_MNIST, D_MNIST))
output_obs = Observable(zeros(MNIST_DEMO_FT, 10))
label_obs = Observable(-1)
selected_digit = Observable(0)
status_obs = Observable(mnist_status(graph, label_obs[], false))

mnist_digit_label(digit) = "Digit $digit"
mnist_temperature_label(value) = @sprintf("%.4g", value)
mnist_rate_label(value) = "$(round(value, digits = 2))"

function mnist_process_running()
    return any(MNIST_DEMO_PROCESSES.isrunning, InteractiveIsing.processes(graph))
end

function dispatch_mnist_index!(idx)
    MNIST_DEMO_WINDOWS._request_graph_process_close!(graph)

    x = copy(@view xdata[:, idx])
    label = labels[idx]

    InteractiveIsing.state(graph[1]) .= reshape(x, size(InteractiveIsing.state(graph[1])))
    InteractiveIsing.off!(graph.index_set, 1)

    image_obs[] = normalize_mnist_for_display(x)
    label_obs[] = label
    process_ref[] = InteractiveIsing.createProcess(graph, dynamics)
    status_obs[] = mnist_status(graph, label, true)
    return nothing
end

function dispatch_selected_digit!()
    digit = selected_digit[]
    idxs = digit_indices[digit + 1]
    isempty(idxs) && return nothing
    return dispatch_mnist_index!(rand(rng, idxs))
end

function dispatch_random_mnist!()
    return dispatch_mnist_index!(rand(rng, axes(xdata, 2)))
end

function mnist_digit_slider_changed(value)
    selected_digit[] = Int(round(value))
    return nothing
end

function mnist_digit_button_clicked(_)
    dispatch_selected_digit!()
    return nothing
end

function mnist_random_button_clicked(_)
    dispatch_random_mnist!()
    return nothing
end

function mnist_temperature_changed(value)
    MNIST_DEMO_WINDOWS._set_temperature!(graph, MNIST_DEMO_FT(value))
    status_obs[] = mnist_status(graph, label_obs[], mnist_process_running())
    return nothing
end

function mnist_frame!(_)
    output_obs[] = collect(vec(InteractiveIsing.state(graph[end])))
    status_obs[] = mnist_status(graph, label_obs[], mnist_process_running())
    return nothing
end

function mnist_close!(_)
    MNIST_DEMO_WINDOWS._request_graph_process_close!(graph)
    return nothing
end

host = window(
    title = "Interactive MNIST Debug",
    size = (1250, 900),
    fps = 30,
    polling_rate = 10,
)

controls = GridLayout(host.figure[1, 1])
digit_slider = Slider(controls[1, 1], range = 0:9, startvalue = selected_digit[], width = 220)
digit_label = Label(controls[1, 2], lift(mnist_digit_label, selected_digit); width = 70)
digit_button = Button(controls[1, 3], label = "Load Digit", width = 120, height = 34)
random_button = Button(controls[1, 4], label = "Random Any", width = 120, height = 34)
Label(controls[1, 5], status_obs; tellwidth = false, halign = :left)
Label(controls[2, 1], "T"; tellwidth = false, halign = :right)
temperature_slider = Slider(
    controls[2, 2:4],
    range = 0:MNIST_DEMO_TEMP_SLIDER_STEP:MNIST_DEMO_TEMP_SLIDER_MAX,
    startvalue = MNIST_DEMO_TEMP,
    width = 360,
)
Label(controls[2, 5], lift(mnist_temperature_label, temperature_slider.value); tellwidth = false, halign = :left)
ups, upsps = MNIST_DEMO_WINDOWS._steps_per_second_observables!(host, graph)
Label(controls[3, 1:2], "Steps per second"; tellwidth = false, halign = :left)
Label(controls[3, 3], lift(mnist_rate_label, ups); tellwidth = false, halign = :left)
Label(controls[3, 4], "per unit"; tellwidth = false, halign = :right)
Label(controls[3, 5], lift(mnist_rate_label, upsps); tellwidth = false, halign = :left)

image_axis = Axis(
    host.figure[3, 1],
    title = "input image",
    aspect = DataAspect(),
)
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

layer_panel = panel!(
    host,
    AllLayersViewPanel(
        graph;
        colormap = :thermal,
        labels = true,
        display_sizes = (
            (70, 70),
            (420, 420),
            (70, 70),
        ),
        axis_kwargs = (
            title = "MNIST 28^2 -> $(MNIST_DEMO_HIDDEN) -> 10",
            aspect = nothing,
        ),
    ),
    (2, 1:2),
)

register!(host, on(mnist_digit_slider_changed, digit_slider.value))
register!(host, on(mnist_digit_button_clicked, digit_button.clicks))
register!(host, on(mnist_random_button_clicked, random_button.clicks))
register!(host, on(mnist_temperature_changed, temperature_slider.value))
register_frame!(host, mnist_frame!)
onclose!(host, mnist_close!)

dispatch_random_mnist!()
