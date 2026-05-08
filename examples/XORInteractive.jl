using InteractiveIsing
using InteractiveIsing.Processes
using InteractiveIsing.Windows
using GLMakie
using LinearAlgebra

const XORFloat = Float64
const XOR_CASES = ((false, false), (false, true), (true, false), (true, true))

xor_case_index(a::Bool, b::Bool) = (a ? 2 : 0) + (b ? 1 : 0) + 1
xor_target(a::Bool, b::Bool) = xor(a, b) ? XORFloat[-1, 1] : XORFloat[1, -1]

function xor_input(a::Bool, b::Bool)
    input = fill(-one(XORFloat), 2, 2)
    input[xor_case_index(a, b)] = one(XORFloat)
    return input
end

function push_symmetric!(rows, cols, vals, i, j, value)
    push!(rows, i)
    push!(cols, j)
    push!(vals, value)
    push!(rows, j)
    push!(cols, i)
    push!(vals, value)
    return nothing
end

function xor_rbm_adjacency(; input_hidden_scale = 1.2, output_field = 3.0)
    rows = Int32[]
    cols = Int32[]
    vals = XORFloat[]

    # Four hidden units are assigned to each one-hot input case. The ideal hidden
    # code is +1 for the active case block and -1 elsewhere.
    for case_idx in 1:4
        for copy_idx in 1:4
            hidden_idx = 4 + 4 * (case_idx - 1) + copy_idx
            for input_idx in 1:4
                weight = input_idx == case_idx ? input_hidden_scale : -input_hidden_scale
                push_symmetric!(rows, cols, vals, input_idx, hidden_idx, weight)
            end
        end
    end

    hidden_code = fill(-one(XORFloat), 16, 4)
    for case_idx in 1:4
        hidden_code[(4 * case_idx - 3):(4 * case_idx), case_idx] .= one(XORFloat)
    end

    target_fields = Matrix{XORFloat}(undef, 4, 2)
    for (case_idx, case) in enumerate(XOR_CASES)
        target_fields[case_idx, :] .= output_field .* xor_target(case...)
    end

    # Least-norm hidden-to-output weights whose ideal hidden codes create the
    # requested output fields.
    output_weights = hidden_code * inv(hidden_code' * hidden_code) * target_fields
    for hidden_local_idx in 1:16
        hidden_idx = 4 + hidden_local_idx
        push_symmetric!(rows, cols, vals, hidden_idx, 21, output_weights[hidden_local_idx, 1])
        push_symmetric!(rows, cols, vals, hidden_idx, 22, output_weights[hidden_local_idx, 2])
    end

    return InteractiveIsing.UndirectedAdjacency(rows, cols, vals, 22, 22; fastwrite = true)
end

function xor_rbm_graph()
    input_layer = Layer(
        2, 2,
        StateSet(-one(XORFloat), one(XORFloat)),
        Continuous(),
        Coords(0, 0, 0);
        periodic = false,
    )
    hidden_layer = Layer(
        4, 4,
        StateSet(-one(XORFloat), one(XORFloat)),
        Continuous(),
        Coords(0, 1, 0);
        periodic = false,
    )
    output_layer = Layer(
        1, 2,
        StateSet(-one(XORFloat), one(XORFloat)),
        Continuous(),
        Coords(0, 4, 0);
        periodic = false,
    )

    b = zeros(XORFloat, 22)
    b[5:20] .= -1.2

    g = IsingGraph(
        input_layer,
        hidden_layer,
        output_layer,
        Bilinear() + MagField(b = b);
        precision = XORFloat,
        adj = xor_rbm_adjacency(),
        index_set = g -> ToggledIndexSet(g),
        initial_state = 0,
    )
    temp!(g, 0.001)
    InteractiveIsing.off!(g.index_set, 1)
    return g
end

function apply_xor_case!(g, a::Bool, b::Bool)
    resetstate!(g)
    state(g[1]) .= xor_input(a, b)
    InteractiveIsing.off!(g.index_set, 1)
    return g
end

function output_status(g, a::Bool, b::Bool)
    out = collect(vec(state(g[3])))
    target = xor_target(a, b)
    prediction = out[2] > out[1]
    mse = sum(abs2, out .- target) / length(target)
    return string(
        "input = (", Int(a), ", ", Int(b), ")    xor = ", Int(xor(a, b)),
        "    output = [", round(out[1]; digits = 3), ", ", round(out[2]; digits = 3), "]",
        "    prediction = ", Int(prediction),
        "    MSE = ", round(mse; digits = 4),
    )
end

function restart_xor_process!(g, dynamics, a::Bool, b::Bool)
    Processes.close(g)
    apply_xor_case!(g, a, b)
    return createProcess(g, dynamics)
end

function xor_interactive()
    g = xor_rbm_graph()
    dynamics = BlockLangevin(stepsize = 0.05, adjusted = false, block_size = 8)

    bit_a = Observable(false)
    bit_b = Observable(false)
    status = Observable(output_status(g, bit_a[], bit_b[]))

    host = window(title = "Interactive XOR RBM", size = (1200, 800), fps = 30, polling_rate = 10)
    controls = GridLayout(host.figure[1, 1])
    host.figure[2, 1] = GridLayout()

    button_a = Button(
        controls[1, 1],
        label = lift(a -> "x1 = $(Int(a))", bit_a),
        width = 110,
        height = 34,
    )
    button_b = Button(
        controls[1, 2],
        label = lift(b -> "x2 = $(Int(b))", bit_b),
        width = 110,
        height = 34,
    )
    Label(controls[1, 3], status; tellwidth = false, halign = :left)

    layer_panel = panel!(
        host,
        AllLayersViewPanel(
            g;
            colormap = :balance,
            labels = true,
            axis_kwargs = (title = "XOR layers: input | hidden | output",),
        ),
        host.figure[2, 1],
    )

    process_ref = Ref{Any}(restart_xor_process!(g, dynamics, bit_a[], bit_b[]))

    function set_case!(a::Bool, b::Bool)
        bit_a[] = a
        bit_b[] = b
        process_ref[] = restart_xor_process!(g, dynamics, a, b)
        status[] = output_status(g, a, b)
        return nothing
    end

    register!(host, on(button_a.clicks) do _
        set_case!(!bit_a[], bit_b[])
    end)
    register!(host, on(button_b.clicks) do _
        set_case!(bit_a[], !bit_b[])
    end)
    register!(host, on(host.open) do isopen
        isopen || Processes.close(g)
    end)
    register_frame!(host) do _
        status[] = output_status(g, bit_a[], bit_b[])
        return nothing
    end

    return (; graph = g, host, layer_panel, process = process_ref)
end

if get(ENV, "INTERACTIVE_XOR_HEADLESS", "false") == "true"
    xor_demo = (; graph = xor_rbm_graph())
else
    xor_demo = xor_interactive()
end
