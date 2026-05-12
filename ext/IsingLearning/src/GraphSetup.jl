export ReducedBoltzmannArchitecture, GraphFromSource, GraphFromInit

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
