export ReducedBoltzmannArchitecture, GraphFromSource, GraphFromInit,
       D_MNIST, MNIST_INPUT_DIM, MNIST_DEFAULT_HIDDEN, MNISTArchitecture, MNISTLayer

const D_MNIST = 28
const MNIST_INPUT_DIM = D_MNIST^2
const MNIST_DEFAULT_HIDDEN = 10 * MNIST_INPUT_DIM

function ReducedBoltzmannArchitecture(layer_sizes...; precision = Float32, b = nothing, weight_generator = nothing)
    layer_gen = [Layer(
            layer_sizes[i],
            Continuous(),
            Coords(0, i, 0)) for i in 1:length(layer_sizes)]
    
    weight_generator = isnothing(weight_generator) ? AllToAllWeightGenerator() : weight_generator
    weight_generators = [deepcopy(weight_generator) for _ in 1:(length(layer_sizes)-1)]

    layers_and_wgs = Any[layer_gen[1]]
    for i in eachindex(weight_generators)
        push!(layers_and_wgs, weight_generators[i], layer_gen[i + 1])
    end

    
    bias = isnothing(b) ? (g -> InteractiveIsing.filltype(Vector, zero(precision), statelen(g))) : b
    clamping_target = g -> InteractiveIsing.filltype(Vector, zero(precision), statelen(g))
    clamping_beta = InteractiveIsing.UniformArray(zero(precision))

    IsingGraph(layers_and_wgs...,
                Ising(b = bias) + Clamping(β = clamping_beta, y = clamping_target);
                index_set = g -> ToggledIndexSet(g))
end

function _mnist_weight_generator(precision, weight_scale, rng)
    scale = precision(weight_scale)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> scale * randn(rng, precision))
end

function _mnist_hidden_shape(hidden::Integer)
    side = floor(Int, sqrt(hidden))
    while side > 1 && hidden % side != 0
        side -= 1
    end
    return (side, hidden ÷ side)
end

"""
    MNISTArchitecture(; hidden = MNIST_DEFAULT_HIDDEN, precision = Float32, weight_scale = 0.01, rng = MersenneTwister(0), weight_generator = nothing)

Construct the single-vector MNIST graph architecture:

`D_MNIST^2 -> hidden -> 10`, with `D_MNIST == 28`. The layers are shaped
for interactive display while keeping those flattened unit counts.
"""
function MNISTArchitecture(;
    hidden::Integer = MNIST_DEFAULT_HIDDEN,
    precision = Float32,
    weight_scale::Real = 0.01,
    rng = MersenneTwister(0),
    weight_generator = nothing,
)
    hidden > 0 || throw(ArgumentError("hidden must be positive, got $(hidden)"))
    wg = isnothing(weight_generator) ? _mnist_weight_generator(precision, weight_scale, rng) : weight_generator
    hidden_rows, hidden_cols = _mnist_hidden_shape(Int(hidden))

    input_layer = Layer(
        D_MNIST,
        D_MNIST,
        StateSet(-one(precision), one(precision)),
        Continuous(),
        Coords(0, 0, 0);
        periodic = false,
    )
    hidden_layer = Layer(
        hidden_rows,
        hidden_cols,
        StateSet(-one(precision), one(precision)),
        Continuous(),
        Coords(0, D_MNIST + 2, 0);
        periodic = false,
    )
    output_layer = Layer(
        2,
        5,
        StateSet(-one(precision), one(precision)),
        Continuous(),
        Coords(0, D_MNIST + hidden_cols + 4, 0);
        periodic = false,
    )

    bias = g -> InteractiveIsing.filltype(Vector, zero(precision), statelen(g))
    clamping_target = g -> InteractiveIsing.filltype(Vector, zero(precision), statelen(g))
    clamping_beta = InteractiveIsing.UniformArray(zero(precision))

    return IsingGraph(
        input_layer,
        wg,
        hidden_layer,
        deepcopy(wg),
        output_layer,
        Bilinear() + MagField(b = bias) + Clamping(β = clamping_beta, y = clamping_target);
        index_set = g -> ToggledIndexSet(g),
    )
end

"""
    MNISTLayer(; graph = MNISTArchitecture(), kwargs...)

Create a `LayeredIsingGraphLayer` around `MNISTArchitecture`, using the first
layer as the flattened image input and the last layer as the ten-class output.
"""
function MNISTLayer(;
    graph = nothing,
    hidden::Integer = MNIST_DEFAULT_HIDDEN,
    precision = Float32,
    weight_scale::Real = 0.01,
    rng = MersenneTwister(0),
    weight_generator = nothing,
    β::Real = precision(0.1),
    fullsweeps::Integer = 1,
    relaxation_steps::Union{Nothing,Integer} = nothing,
    free_relaxation_steps::Union{Nothing,Integer} = relaxation_steps,
    nudged_relaxation_steps::Union{Nothing,Integer} = relaxation_steps,
    dynamics_algorithm = Metropolis(),
    nudged_dynamics_algorithm = deepcopy(dynamics_algorithm),
    validation_algorithm = deepcopy(dynamics_algorithm),
)
    graph = isnothing(graph) ? MNISTArchitecture(;
        hidden,
        precision,
        weight_scale,
        rng,
        weight_generator,
    ) : graph

    return LayeredIsingGraphLayer(
        graph;
        input_idxs = layerrange(graph[1]),
        output_idxs = layerrange(graph[end]),
        β,
        fullsweeps,
        relaxation_steps,
        free_relaxation_steps,
        nudged_relaxation_steps,
        dynamics_algorithm,
        nudged_dynamics_algorithm,
        validation_algorithm,
    )
end

"""
    Create a graph copy, with separate state, but shared data
"""
function GraphFromSource(g::IsingGraph; init! = identity)
    gnew = IsingGraph(
        copy(state(g)),
        adj(g),
        temp(g),
        g.default_algorithm,
        g.hamiltonian,
        g.index_set,
        g.addons,
        g.layers,
    )
    init!(gnew)
    return gnew
end

function GraphFromInit(g::IsingGraph, parameters; init! = identity)
    colptrs = getcolptrs(adj(g))
    rowvals = getrowvals(adj(g))
    nzvals = parameters.weights
    new_adj = UndirectedAdjacency(colptrs, rowvals, nzvals, size(adj(g)), diag = parameters.α_i)
    gnew = IsingGraph(
        copy(state(g)),
        new_adj,
        temp(g),
        g.default_algorithm,
        Ising(b = parameters.biases) + Clamping(β = InteractiveIsing.UniformArray(zero(eltype(g))), y = g -> InteractiveIsing.filltype(Vector, zero(eltype(g)), statelen(g))),
        g.index_set,
        g.addons,
        g.layers,
    )
    init!(gnew)
    return gnew
end
