using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using InteractiveIsing.Windows
using GLMakie
using Printf

# Interactive demo for the honest 2-input XOR experiment:
# input bits -> 16 hidden spins -> 2 output spins.
#
# The parameters are hardcoded from an earlier learning run that used discrete
# spins, Metropolis dynamics, T = 0.5, β = 1.0,
# hidden = 16, and reached repeated-state MSE ≈ 0.0024 with accuracy 1.0.

const XORFloat = Float64

xor_target(a::Bool, b::Bool) = xor(a, b) ? XORFloat[-1, 1] : XORFloat[1, -1]
xor_input(a::Bool, b::Bool) = reshape(XORFloat[a ? 1 : -1, b ? 1 : -1], 1, 2)

const XOR_TRAINED_ROWS = Int32[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 1, 2, 19, 20, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
const XOR_TRAINED_COLS = Int32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
const XOR_TRAINED_VALS = XORFloat[-0.5604433996826013, -0.6149269568283259, -0.39065105748149304, 0.02572723207768589, 0.6878419884330677, -0.9437288018600382, 1.002338477092773, -0.9407269052604983, 0.7516460363012676, -0.9783003387608045, -0.9082665619564819, 0.9656862236298422, -1.0782671605197574, 0.11650913953320273, 0.559208889594522, -0.7948990047263502, 0.5425509282359142, -0.6702915864705159, 0.5021584087708126, 0.2557750375236546, -0.6359848235649086, 0.9592980926121469, 0.9746674003106934, 0.9178738496903985, 0.8005974526596435, -0.9659098349292364, -0.9834351564726124, -0.9393631159907598, 1.1063328056698831, 0.0302943544335638, -0.544650672405431, -0.76179147019087, -0.5604433996826013, 0.5425509282359142, -0.3438623628806701, 0.22870458912734004, -0.6149269568283259, -0.6702915864705159, -0.3552619461091675, 0.3753763922750342, -0.39065105748149304, 0.5021584087708126, 0.3359500906727113, -0.3285121148375531, 0.02572723207768589, 0.2557750375236546, 0.641565284866686, -0.47296165765309617, 0.6878419884330677, -0.6359848235649086, 0.2731695161895233, -0.3935605576707883, -0.9437288018600382, 0.9592980926121469, -0.41212878578754647, 0.5910266204836354, 1.002338477092773, 0.9746674003106934, -0.6049690573450767, 0.456407639950499, -0.9407269052604983, 0.9178738496903985, 0.5568457603748477, -0.45273013198672696, 0.7516460363012676, 0.8005974526596435, 0.47984423505021084, -0.40939614494437465, -0.9783003387608045, -0.9659098349292364, 0.5767327056698859, -0.4485838248466105, -0.9082665619564819, -0.9834351564726124, -0.5272560144897214, 0.5310311421289996, 0.9656862236298422, -0.9393631159907598, 0.6206442717004053, -0.3320214135918956, -1.0782671605197574, 1.1063328056698831, 0.5243198950277816, -0.6743376376208123, 0.11650913953320273, 0.0302943544335638, -0.1771936775270266, -0.13642147753912973, 0.559208889594522, -0.544650672405431, 0.38582631482786994, -0.2719685791102587, -0.7948990047263502, -0.76179147019087, 0.35144992808922876, -0.5544026254391693, -0.3438623628806701, -0.3552619461091675, 0.3359500906727113, 0.641565284866686, 0.2731695161895233, -0.41212878578754647, -0.6049690573450767, 0.5568457603748477, 0.47984423505021084, 0.5767327056698859, -0.5272560144897214, 0.6206442717004053, 0.5243198950277816, -0.1771936775270266, 0.38582631482786994, 0.35144992808922876, 0.22870458912734004, 0.3753763922750342, -0.3285121148375531, -0.47296165765309617, -0.3935605576707883, 0.5910266204836354, 0.456407639950499, -0.45273013198672696, -0.40939614494437465, -0.4485838248466105, 0.5310311421289996, -0.3320214135918956, -0.6743376376208123, -0.13642147753912973, -0.2719685791102587, -0.5544026254391693]
const XOR_TRAINED_BIAS = XORFloat[-0.03733438975151456, 0.005456473972866237, -0.5869931750099047, 0.6356437672785586, 0.38514570483978566, -0.07424934497893682, 0.661881484196031, -0.945865440732354, 0.9553156163170093, 0.9041407841563515, -0.7941617691202199, -0.9585002586457465, 0.9158657406847038, 0.9460159929885779, 1.0243522897042012, 0.13684591533946133, 0.6122608854576082, -0.7512375025046275, -0.0681651504612854, 0.14536138441554353]

function xor_trained_adjacency()
    return InteractiveIsing.UndirectedAdjacency(
        XOR_TRAINED_ROWS,
        XOR_TRAINED_COLS,
        XOR_TRAINED_VALS,
        20,
        20;
        fastwrite = true,
    )
end

function xor_rbm_graph()
    input_layer = Layer(
        1, 2,
        StateSet(-one(XORFloat), one(XORFloat)),
        Continuous(),
        Coords(0, 0, 0);
        periodic = false,
    )
    hidden_layer = Layer(
        4, 4,
        StateSet(-one(XORFloat), one(XORFloat)),
        Continuous(),
        Coords(0, 3, 0);
        periodic = false,
    )
    output_layer = Layer(
        1, 2,
        StateSet(-one(XORFloat), one(XORFloat)),
        Continuous(),
        Coords(0, 8, 0);
        periodic = false,
    )

    g = IsingGraph(
        input_layer,
        hidden_layer,
        output_layer,
        Bilinear() + MagField(b = copy(XOR_TRAINED_BIAS));
        precision = XORFloat,
        adj = xor_trained_adjacency(),
        index_set = g -> ToggledIndexSet(g),
    )
    temp!(g, 0.5)
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
    return @sprintf(
        "input = (%d, %d)    xor = %d    output = [% .3f, % .3f]    prediction = %d    MSE = %.4f",
        Int(a),
        Int(b),
        Int(xor(a, b)),
        out[1],
        out[2],
        Int(prediction),
        mse,
    )
end

function restart_xor_process!(g, dynamics, a::Bool, b::Bool)
    StatefulAlgorithms.close(g)
    apply_xor_case!(g, a, b)
    return createProcess(g, dynamics)
end

function xor_interactive()
    g = xor_rbm_graph()
    dynamics = LocalLangevin()

    bit_a = Observable(false)
    bit_b = Observable(false)
    status = Observable(output_status(g, bit_a[], bit_b[]))

    host = window(title = "Interactive XOR RBM Demo", size = (1200, 800), fps = 30, polling_rate = 10)
    controls = GridLayout(host.figure[1, 1])

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
            axis_kwargs = (title = "2-bit XOR: input | hidden | output",),
        ),
        (2, 1),
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
