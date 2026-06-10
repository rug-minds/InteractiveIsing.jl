using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using InteractiveIsing.Windows
using GLMakie
using Printf

# Full-graph temperature probe for the learned scalar XOR system.
#
# Architecture shown here:
#   2 input spins -> 2x2 hidden spins -> 1 output spin
#
# The weights are hardwired from an earlier learning run. That graph is not
# loaded here; this example is standalone in InteractiveIsing.

const XOR_TEMP_INITIAL = 0.005
const XOR_TEMP_SLIDER_MAX = 0.05
const XOR_TEMP_SLIDER_STEP = 0.0005
const XOR_TEMP_STEPSIZE = 0.4
const XOR_TEMP_ADJUSTED = false

const XOR_SCALAR_ROWS = Int32[
    3, 4, 5, 6,
    3, 4, 5, 6,
    1, 2, 7,
    1, 2, 7,
    1, 2, 7,
    1, 2, 7,
    3, 4, 5, 6,
]
const XOR_SCALAR_COLS = Int32[
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7, 7,
]
const XOR_SCALAR_VALS = Float64[
    -2.059360965020584,
    -2.3444136295957,
    -2.6962017356407846,
    2.4455501169952716,
    1.3666889883317106,
    -1.9291888100191303,
    -2.20517474388871,
    1.856543830061987,
    -2.059360965020584,
    1.3666889883317106,
    -1.5875683574809618,
    -2.3444136295957,
    -1.9291888100191303,
    -0.7319416994303141,
    -2.6962017356407846,
    -2.20517474388871,
    -1.5521875180500744,
    2.4455501169952716,
    1.856543830061987,
    1.0769611940999628,
    -1.5875683574809618,
    -0.7319416994303141,
    -1.5521875180500744,
    1.0769611940999628,
]
const XOR_SCALAR_BIAS = Float64[
    0.030970382132197858,
    -0.01874469691425061,
    1.6071298224099542,
    -0.8868806686012309,
    -1.331763791276463,
    1.1919940478467983,
    -0.7662792894953001,
]

"""Return the scalar XOR target used by the `2 -> 4 -> 1` runs."""
xor_scalar_target(a::Bool, b::Bool) = xor(a, b) ? 1.0 : -1.0

"""Return the bipolar two-spin input vector for the selected XOR case."""
xor_scalar_input(a::Bool, b::Bool) = reshape(Float64[a ? 1 : -1, b ? 1 : -1], 1, 2)

"""Return the learned seven-spin scalar XOR adjacency."""
xor_scalar_adjacency() = InteractiveIsing.UndirectedAdjacency(XOR_SCALAR_ROWS, XOR_SCALAR_COLS, XOR_SCALAR_VALS, 7, 7; fastwrite = true)

"""
    xor_temperature_graph()

Build the display-friendly `1x2 -> 2x2 -> 1x1` graph using the hardwired
learned adjacency and bias from the scalar XOR run.
"""
function xor_temperature_graph()
    input_layer = Layer(
        1, 2,
        StateSet(-1.0, 1.0),
        Continuous(),
        Coords(0, 0, 0);
        periodic = false,
    )
    hidden_layer = Layer(
        2, 2,
        StateSet(-1.0, 1.0),
        Continuous(),
        Coords(0, 3, 0);
        periodic = false,
    )
    output_layer = Layer(
        1, 1,
        StateSet(-1.0, 1.0),
        Continuous(),
        Coords(0, 7, 0);
        periodic = false,
    )

    graph = IsingGraph(
        input_layer,
        hidden_layer,
        output_layer,
        Bilinear() + MagField(b = copy(XOR_SCALAR_BIAS));
        precision = Float64,
        adj = xor_scalar_adjacency(),
        index_set = g -> ToggledIndexSet(g),
    )
    temp!(graph, XOR_TEMP_INITIAL)
    InteractiveIsing.off!(graph.index_set, 1)
    return graph
end

"""
    apply_xor_temperature_case!(graph, a, b)

Reset the graph, write the two input spins, and keep the input layer out of the
sampler. Hidden and output spins are reinitialized by `resetstate!`.
"""
function apply_xor_temperature_case!(graph, a::Bool, b::Bool)
    resetstate!(graph)
    state(graph[1]) .= xor_scalar_input(a, b)
    InteractiveIsing.off!(graph.index_set, 1)
    return graph
end

"""Format the live scalar output, target, prediction, MSE, and temperature."""
function xor_temperature_status(graph, a::Bool, b::Bool)
    out = only(vec(state(graph[3])))
    target = xor_scalar_target(a, b)
    prediction = out > zero(out)
    mse = abs2(out - target)
    return @sprintf(
        "input=(%d,%d) xor=%d output=% .4f target=% .1f prediction=%d MSE=%.5f T=%.5g",
        Int(a),
        Int(b),
        Int(xor(a, b)),
        out,
        target,
        Int(prediction),
        mse,
        temp(graph),
    )
end

"""Restart the running process after changing input or resetting the state."""
function restart_xor_temperature_process!(graph, dynamics, a::Bool, b::Bool)
    InteractiveIsing.Windows._request_graph_process_close!(graph)
    apply_xor_temperature_case!(graph, a, b)
    return createProcess(graph, dynamics)
end

"""
    xor_temperature_slider!(host, cell, graph)

Add a fine-grained temperature slider. It writes through the window helper so
running process-context temperature fields are updated together with `temp(g)`.
"""
function xor_temperature_slider!(host, cell, graph)
    grid = GridLayout(cell)
    Label(grid[1, 1], "T", width = 24, tellwidth = false)
    slider = Slider(
        grid[1, 2],
        range = 0.0:XOR_TEMP_SLIDER_STEP:XOR_TEMP_SLIDER_MAX,
        value = temp(graph),
        horizontal = true,
        width = 320,
    )
    Label(
        grid[1, 3],
        lift(x -> @sprintf("%.5g", x), slider.value),
        width = 80,
        tellwidth = false,
    )
    InteractiveIsing.Windows.register!(host, on(slider.value) do value
        InteractiveIsing.Windows._set_temperature!(graph, value)
        return nothing
    end)
    return slider
end

"""
    xor_temperature_stability()

Open the interactive temperature-stability probe for the `2 -> 2x2 -> 1`
learned XOR graph.
"""
function xor_temperature_stability()
    graph = xor_temperature_graph()
    dynamics = LocalLangevin(
        stepsize = XOR_TEMP_STEPSIZE,
        adjusted = XOR_TEMP_ADJUSTED,
        order = :random,
        group_steps = 1,
    )

    bit_a = Observable(false)
    bit_b = Observable(false)
    running = Observable(true)
    status = Observable(xor_temperature_status(graph, bit_a[], bit_b[]))

    host = InteractiveIsing.Windows.window(
        title = "XOR 2->2x2->1 Langevin Temperature Stability",
        size = (1250, 850),
        fps = 30,
        polling_rate = 10,
    )
    controls = GridLayout(host.figure[1, 1])

    button_a = Button(controls[1, 1], label = lift(a -> "x1 = $(Int(a))", bit_a), width = 110, height = 34)
    button_b = Button(controls[1, 2], label = lift(b -> "x2 = $(Int(b))", bit_b), width = 110, height = 34)
    reset_button = Button(controls[1, 3], label = "Reset", width = 110, height = 34)
    pause_button = Button(controls[1, 4], label = lift(x -> x ? "Pause" : "Run", running), width = 110, height = 34)
    slider = xor_temperature_slider!(host, controls[1, 5], graph)
    Label(controls[2, 1:5], status; tellwidth = false, halign = :left)

    panel = InteractiveIsing.Windows.panel!(
        host,
        InteractiveIsing.Windows.AllLayersViewPanel(
            graph;
            colormap = :balance,
            labels = true,
            axis_kwargs = (title = "scalar XOR: input 1x2 | hidden 2x2 | output 1x1",),
        ),
        (2, 1),
    )

    process_ref = Ref{Any}(restart_xor_temperature_process!(graph, dynamics, bit_a[], bit_b[]))

    function set_case!(a::Bool, b::Bool)
        bit_a[] = a
        bit_b[] = b
        process_ref[] = restart_xor_temperature_process!(graph, dynamics, a, b)
        InteractiveIsing.Windows._set_temperature!(graph, slider.value[])
        running[] = true
        status[] = xor_temperature_status(graph, a, b)
        return nothing
    end

    InteractiveIsing.Windows.register!(host, on(button_a.clicks) do _
        set_case!(!bit_a[], bit_b[])
    end)
    InteractiveIsing.Windows.register!(host, on(button_b.clicks) do _
        set_case!(bit_a[], !bit_b[])
    end)
    InteractiveIsing.Windows.register!(host, on(reset_button.clicks) do _
        set_case!(bit_a[], bit_b[])
    end)
    InteractiveIsing.Windows.register!(host, on(pause_button.clicks) do _
        if running[]
            StatefulAlgorithms.close(graph)
            running[] = false
        else
            process_ref[] = createProcess(graph, dynamics)
            InteractiveIsing.Windows._set_temperature!(graph, slider.value[])
            running[] = true
        end
        return nothing
    end)
    InteractiveIsing.Windows.register_frame!(host) do _
        status[] = xor_temperature_status(graph, bit_a[], bit_b[])
        return nothing
    end
    InteractiveIsing.Windows.onclose!(host) do _
        InteractiveIsing.Windows._request_graph_process_close!(graph)
    end

    return (; graph, host, panel, process = process_ref, slider, bit_a, bit_b, dynamics)
end

xor_temperature_demo = xor_temperature_stability()
