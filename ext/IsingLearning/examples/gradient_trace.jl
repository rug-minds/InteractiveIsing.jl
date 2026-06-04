using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Optimisers
using Random
using SparseArrays
using LinearAlgebra

Random.seed!(1234)

const RELAXATION_STEPS = parse(Int, get(ENV, "ISING_TRACE_RELAXATION_STEPS", "10"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_TRACE_WEIGHT_SCALE", "0.05"))

function small_weight_generator(seed::Integer = 4321)
    rng = Random.MersenneTwister(seed)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> WEIGHT_SCALE * randn(rng, Float32))
end

function pattern_vertical()
    p = Matrix{Float32}(undef, 4, 4)
    for row in 1:4, col in 1:4
        p[row, col] = col <= 2 ? -1f0 : 1f0
    end
    return vec(p)
end

function pattern_horizontal()
    p = Matrix{Float32}(undef, 4, 4)
    for row in 1:4, col in 1:4
        p[row, col] = row <= 2 ? -1f0 : 1f0
    end
    return vec(p)
end

function xor_input(a::Bool, b::Bool)
    pv = pattern_vertical()
    ph = pattern_horizontal()
    return 0.5f0 .* ((a ? pv : -pv) .+ (b ? ph : -ph))
end

xor_target(a::Bool, b::Bool) = xor(a, b) ? Float32[-1, 1] : Float32[1, -1]

function finite_checks(name, x)
    all(isfinite, x) || error("$name contains non-finite values")
    return nothing
end

function fixed_state_energy(g, s)
    h = g.hamiltonian
    poly = h[InteractiveIsing.PolynomialHamiltonian]
    mag = h[InteractiveIsing.MagField]
    J = adj(g)
    return Float64(
        -0.5f0 * dot(s, J * s) -
        mag.c * dot(mag.b, s) +
        sum(poly.c[] .* poly.lp .* (s .^ 2))
    )
end

function central_difference!(g, params, field::Symbol, idx::Int, s; ε = 1f-3)
    plus = deepcopy(params)
    minus = deepcopy(params)
    getproperty(plus, field)[idx] += ε
    getproperty(minus, field)[idx] -= ε

    IsingLearning.sync_graph_params!(g, plus)
    eplus = fixed_state_energy(g, s)
    IsingLearning.sync_graph_params!(g, minus)
    eminus = fixed_state_energy(g, s)
    IsingLearning.sync_graph_params!(g, params)

    return (eplus - eminus) / (2Float64(ε))
end

function trace_one_sample!()
    graph = ReducedBoltzmannArchitecture(16, 8, 2; precision = Float32, weight_generator = small_weight_generator())
    dynamics = GlobalLangevin(stepsize = 1f-3, adjusted = false, group_steps = 1)
    layer = LayeredIsingGraphLayer(
        () -> ReducedBoltzmannArchitecture(16, 8, 2; precision = Float32, weight_generator = small_weight_generator());
        input_idxs = layerrange(graph[1]),
        output_idxs = layerrange(graph[end]),
        β = 0.1f0,
        fullsweeps = 1,
        relaxation_steps = RELAXATION_STEPS,
        dynamics_algorithm = dynamics,
        validation_algorithm = deepcopy(dynamics),
    )

    trainer = init_mnist_trainer(layer; graph, numthreads = 1, optimiser = Optimisers.Adam(1f-3))
    worker = only(trainer.workers)
    buffer = StatefulAlgorithms.context(worker)._state.buffers

    IsingLearning.zero_buffer!(buffer)
    StatefulAlgorithms.context(worker)._state.x .= xor_input(false, true)
    StatefulAlgorithms.context(worker)._state.y .= xor_target(false, true)
    StatefulAlgorithms.reset!(worker)
    run(worker)
    wait(worker)
    close(worker)

    g = StatefulAlgorithms.context(worker).dynamics.model
    s_plus = Float32.(StatefulAlgorithms.context(worker).plus_capture.captured)
    s_minus = Float32.(StatefulAlgorithms.context(worker).minus_capture.captured)
    finite_checks("s_plus", s_plus)
    finite_checks("s_minus", s_minus)
    finite_checks("raw weight gradient", buffer.w)
    finite_checks("raw bias gradient", buffer.b)
    finite_checks("raw local-potential gradient", buffer.α)

    recomputed = IsingLearning.gradient_buffer(g)
    IsingLearning.contrastive_gradient(g, s_plus, s_minus, layer.β; buffers = recomputed)

    w_err = maximum(abs.(buffer.w .- recomputed.w))
    b_err = maximum(abs.(buffer.b .- recomputed.b))
    α_err = maximum(abs.(buffer.α .- recomputed.α))

    bilinear = g.hamiltonian[InteractiveIsing.Bilinear]
    magfield = g.hamiltonian[InteractiveIsing.MagField]
    polynomial = g.hamiltonian[InteractiveIsing.PolynomialHamiltonian]
    params = IsingLearning.read_graph_params(g)
    plus_w_all = similar(params.w)
    InteractiveIsing.parameter_derivative(bilinear, s_plus; dJ = plus_w_all)
    plus_b_all = InteractiveIsing.parameter_derivative(magfield, s_plus).db
    plus_α_all = InteractiveIsing.parameter_derivative(polynomial, s_plus).dlp
    w_idx = findfirst(x -> abs(x) > 1f-6, plus_w_all)
    isnothing(w_idx) && error("could not find a nonzero weight derivative to trace")
    out_idx = first(layer.output_layer)

    plus_w_fd = central_difference!(g, params, :w, w_idx, s_plus)
    plus_b_fd = central_difference!(g, params, :b, out_idx, s_plus)
    plus_α_fd = central_difference!(g, params, :α, out_idx, s_plus)

    plus_w = plus_w_all[w_idx]
    plus_b = plus_b_all[out_idx]
    plus_α = plus_α_all[out_idx]

    println("Gradient trace for one XOR sample")
    println("  captured states: plus_norm=$(norm(s_plus)), minus_norm=$(norm(s_minus)), diff_norm=$(norm(s_plus .- s_minus))")
    println("  contrastive buffer recompute max errors: ", (w = w_err, b = b_err, α = α_err))
    println("  fixed-state finite differences at s_plus:")
    println("    w[$w_idx]: analytic=$plus_w finite_difference=$plus_w_fd abs_error=$(abs(Float64(plus_w) - plus_w_fd))")
    println("    b[$out_idx]: analytic=$plus_b finite_difference=$plus_b_fd abs_error=$(abs(Float64(plus_b) - plus_b_fd))")
    println("    α[$out_idx]: analytic=$plus_α finite_difference=$plus_α_fd abs_error=$(abs(Float64(plus_α) - plus_α_fd))")
    println("  scaled batch gradient norms: ", (
        w = norm(buffer.w) / (2layer.β),
        b = norm(buffer.b) / (2layer.β),
        α = norm(buffer.α) / (2layer.β),
    ))
    close_trainer!(trainer)
end

trace_one_sample!()
