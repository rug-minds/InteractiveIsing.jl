using Test
using InteractiveIsing
using LinearAlgebra
using SparseArrays

struct MissingFillArray{T,N} <: AbstractArray{T,N} end

struct TemplateLayerHookTerm{P} <: InteractiveIsing.LayerTerm
    layer::Int
    parameters::P
end

function TemplateLayerHookTerm(; layer = 1, c = nothing)
    params = InteractiveIsing.Parameters(
        InteractiveIsing.parameter(;
            c,
            type = Number,
            default = 1,
            ensure = InteractiveIsing.ensure_isinggraph_eltype,
        ),
    )
    return TemplateLayerHookTerm(Int(layer), params)
end

InteractiveIsing._calculate(::InteractiveIsing.d_iH, term::TemplateLayerHookTerm, layer, proposal::InteractiveIsing.SingleSpinProposal) =
    term.c * InteractiveIsing.at_idx(proposal)

InteractiveIsing._calculate(::InteractiveIsing.ΔH, term::TemplateLayerHookTerm, layer, proposal) =
    term.c * getfield(proposal, :at_idx)

struct DirectCalculateLayerTerm <: InteractiveIsing.LayerTerm
    layer::Int
end

InteractiveIsing.calculate(::InteractiveIsing.d_iH, ::DirectCalculateLayerTerm, model::InteractiveIsing.AbstractIsingGraph, proposal::InteractiveIsing.SingleSpinProposal) =
    eltype(model)(99)

@testset "SingleSpinProposal endpoint state API" begin
    g = IsingGraph(2, 1, Continuous(), Quadratic(c = 1, localpotential = [1, 1]); precision = Float64)
    InteractiveIsing.graphstate(g) .= [0.25, -0.5]

    proposal = SingleSpinProposal(1, 0.25, 0.75, 1)
    @test proposal isa FlipProposal
    @test InteractiveIsing.proposed_value(InteractiveIsing.graphstate(g), proposal) == 0.75
    @test InteractiveIsing.proposed_value(InteractiveIsing.graphstate(g), SingleSpinProposal(2, -0.5, NoChange(), 1)) == -0.5

    derivative = InteractiveIsing.calculate(InteractiveIsing.d_iH(), g.hamiltonian, g, proposal)
    @test derivative ≈ 1.5
    @test InteractiveIsing.graphstate(g) == [0.25, -0.5]
end

@testset "Hamiltonian Term Field Access" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)
    hts = g.hamiltonian

    @test InteractiveIsing.getparam(hts, InteractiveIsing.Bilinear, Val(:J)) ===
          InteractiveIsing.gethamiltonian(hts, InteractiveIsing.Bilinear).J

    @test InteractiveIsing.getparam(hts, InteractiveIsing.MagField, Val(:b)) ===
          InteractiveIsing.gethamiltonian(hts, InteractiveIsing.MagField).b

    @test InteractiveIsing.getparam(hts, InteractiveIsing.Bilinear, :J) ===
          InteractiveIsing.gethamiltonian(hts, InteractiveIsing.Bilinear).J

    @test InteractiveIsing.getparam(hts, :J) ===
          InteractiveIsing.getparam(hts, Val(:J))

    duplicate_hts = InteractiveIsing.HamiltonianTerms(
        InteractiveIsing.Bilinear(g),
        InteractiveIsing.Bilinear(g),
    )

    @test_throws ErrorException InteractiveIsing.getparam(
        duplicate_hts,
        InteractiveIsing.Bilinear,
        Val(:J),
    )

    @test InteractiveIsing.gethamiltonianfield(hts, InteractiveIsing.Bilinear, Val(:J)) ===
          InteractiveIsing.getparam(hts, InteractiveIsing.Bilinear, Val(:J))
end

@testset "Hamiltonian Parameter Filling" begin
    default_graph = IsingGraph(2, 2, Continuous(), Ising(); precision = Float32)
    default_b = InteractiveIsing.gethamiltonian(default_graph.hamiltonian, InteractiveIsing.MagField).b
    @test default_b isa InteractiveIsing.ConstFill
    @test eltype(default_b) === Float32
    @test length(default_b) == 4
    @test all(==(0f0), default_b)

    scalar_graph = IsingGraph(2, 2, Continuous(), Ising(b = 1); precision = Float32)
    scalar_b = InteractiveIsing.gethamiltonian(scalar_graph.hamiltonian, InteractiveIsing.MagField).b
    @test scalar_b isa InteractiveIsing.UniformArray
    @test eltype(scalar_b) === Float32
    @test length(scalar_b) == 4
    @test all(==(1f0), scalar_b)
    scalar_b[] = 2f0
    @test all(==(2f0), scalar_b)

    singleton_graph = IsingGraph(2, 2, Continuous(), Ising(b = [1]); precision = Float32)
    singleton_b = InteractiveIsing.gethamiltonian(singleton_graph.hamiltonian, InteractiveIsing.MagField).b
    @test singleton_b isa InteractiveIsing.UniformArray
    @test eltype(singleton_b) === Float32
    @test length(singleton_b) == 4
    @test all(==(1f0), singleton_b)

    vector_type_graph = IsingGraph(2, 2, Continuous(), Ising(b = Vector); precision = Float32)
    vector_type_b = InteractiveIsing.gethamiltonian(vector_type_graph.hamiltonian, InteractiveIsing.MagField).b
    @test vector_type_b isa Vector{Float32}
    @test length(vector_type_b) == 4
    @test all(==(0f0), vector_type_b)

    const_graph = IsingGraph(2, 2, Continuous(), Ising(b = ConstFill(1)); precision = Float32)
    const_b = InteractiveIsing.gethamiltonian(const_graph.hamiltonian, InteractiveIsing.MagField).b
    @test const_b isa InteractiveIsing.ConstFill
    @test eltype(const_b) === Float32
    @test length(const_b) == 4
    @test all(==(1f0), const_b)

    custom_graph = IsingGraph(2, 2, Continuous(), Ising(b = [1, 2, 3, 4]); precision = Float32)
    custom_b = InteractiveIsing.gethamiltonian(custom_graph.hamiltonian, InteractiveIsing.MagField).b
    @test custom_b isa Vector{Float32}
    @test custom_b == Float32[1, 2, 3, 4]

    @test_throws DimensionMismatch IsingGraph(2, 2, Continuous(), Ising(b = [1, 2]); precision = Float32)
    @test_throws ArgumentError IsingGraph(2, 2, Continuous(), Ising(b = MissingFillArray); precision = Float32)
end

@testset "LayerTerm Hook Semantics" begin
    layer1 = Layer(2, Continuous(), StateSet(-1, 1); periodic = false)
    layer2 = Layer(3, Continuous(), StateSet(-1, 1); periodic = false)
    g = IsingGraph(layer1, layer2, TemplateLayerHookTerm(layer = 2, c = 2); precision = Float64)
    hterm = g.hamiltonian

    @test InteractiveIsing.layeridx(hterm) == 2
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, SingleSpinProposal(1, 0.0, NoChange(), 1)) == 0.0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, SingleSpinProposal(4, 0.0, NoChange(), 2)) == 4.0

    outside = FlipProposal(1, 0.0, 1.0, 1)
    inside = FlipProposal(5, 0.0, 1.0, 2)
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, outside) == 0.0
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, inside) == 6.0

    direct = DirectCalculateLayerTerm(2)
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), direct, g, SingleSpinProposal(1, 0.0, NoChange(), 1)) == 99.0
end

@testset "ToLayer Wrapper Semantics" begin
    zero_adj(n) = InteractiveIsing.UndirectedAdjacency(spzeros(Float64, n, n), zeros(Float64, n))

    layer1 = Layer(2, Continuous(), StateSet(-10, 10); periodic = false)
    layer2 = Layer(3, Continuous(), StateSet(-10, 10); periodic = false)
    g = IsingGraph(layer1, layer2, ToLayer(2, MagField(b = 1)); precision = Float64, adj = zero_adj(5))
    InteractiveIsing.graphstate(g) .= Float64[1, 2, 3, 4, 5]

    single = IsingGraph(Layer(3, Continuous(), StateSet(-10, 10); periodic = false), MagField(b = 1); precision = Float64, adj = zero_adj(3))
    InteractiveIsing.graphstate(single) .= Float64[3, 4, 5]

    outside = FlipProposal(1, 1.0, 7.0, 1)
    inside = FlipProposal(4, 4.0, 8.0, 2)
    local_inside = FlipProposal(2, 4.0, 8.0, 1)

    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), g.hamiltonian, g, SingleSpinProposal(1, 1.0, NoChange(), 1)) == 0.0
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, outside) == 0.0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), g.hamiltonian, g, SingleSpinProposal(4, 4.0, NoChange(), 2)) ≈
          InteractiveIsing.calculate(InteractiveIsing.d_iH(), single.hamiltonian, single, SingleSpinProposal(2, 4.0, NoChange(), 1))
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, inside) ≈
          InteractiveIsing.calculate(InteractiveIsing.ΔH(), single.hamiltonian, single, local_inside)

    rows = [4, 1, 3]
    cols = [3, 3, 1]
    vals = Float64[2, 10, 10]
    global_adj = InteractiveIsing.UndirectedAdjacency(sparse(rows, cols, vals, 4, 4), zeros(Float64, 4))
    bilinear_graph = IsingGraph(
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        ToLayer(2, Bilinear());
        precision = Float64,
        adj = global_adj,
    )
    InteractiveIsing.graphstate(bilinear_graph) .= Float64[1, 1, 3, 5]
    @test InteractiveIsing.calculate(
        InteractiveIsing.ΔH(),
        bilinear_graph.hamiltonian,
        bilinear_graph,
        FlipProposal(3, 3.0, 4.0, 2),
    ) ≈ -10.0

    wrapped_ising = IsingGraph(
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        ToLayer(2, Ising(c = ConstVal(0.0), b = 0.5));
        precision = Float64,
        adj = zero_adj(4),
    )
    InteractiveIsing.graphstate(wrapped_ising) .= Float64[1, 2, 3, 4]
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), wrapped_ising.hamiltonian, wrapped_ising, FlipProposal(1, 1.0, 7.0, 1)) == 0.0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), wrapped_ising.hamiltonian, wrapped_ising, SingleSpinProposal(4, 4.0, NoChange(), 2)) ≈ -0.5

    mixed = IsingGraph(
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        MagField(b = 1) + ToLayer(2, MagField(b = 2));
        precision = Float64,
        adj = zero_adj(4),
    )
    InteractiveIsing.graphstate(mixed) .= Float64[1, 2, 3, 4]
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), mixed.hamiltonian, mixed, SingleSpinProposal(1, 1.0, NoChange(), 1)) ≈ -1.0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), mixed.hamiltonian, mixed, SingleSpinProposal(4, 4.0, NoChange(), 2)) ≈ -3.0

    lookup_h = IsingGraph(
        Layer(2, Continuous(), StateSet(-10, 10); periodic = false),
        Layer(3, Continuous(), StateSet(-10, 10); periodic = false),
        MagField(b = 1) + ToLayer(2, Ising(c = ConstVal(0.0), b = 0.5));
        precision = Float64,
        adj = zero_adj(5),
    )
    found = InteractiveIsing.gethamiltonian(lookup_h.hamiltonian, InteractiveIsing.MagField, 2)
    @test found isa InteractiveIsing.MagField
    @test length(found.b) == length(InteractiveIsing.state(lookup_h[2]))
end

@testset "CosineInteraction Semantics" begin
    cosine_adj(weight = 1.0) =
        InteractiveIsing.UndirectedAdjacency(
            sparse([1, 2], [2, 1], Float64[weight, weight], 2, 2),
            zeros(Float64, 2),
        )

    continuous_layer = Layer(2, Continuous(), StateSet(0.0, 1.0); periodic = false)
    g = IsingGraph(continuous_layer, CosineInteraction(); precision = Float64, adj = cosine_adj(1.5))
    InteractiveIsing.graphstate(g) .= [0.0, 0.25]
    hterm = g.hamiltonian

    expected_energy = -1.5 * cos(2π * (0.0 - 0.25))
    @test InteractiveIsing.calculate(InteractiveIsing.H(), hterm, g) ≈ expected_energy

    proposal = FlipProposal(2, 0.25, 0.5, 1)
    before = InteractiveIsing.calculate(InteractiveIsing.H(), hterm, g)
    InteractiveIsing.graphstate(g)[2] = 0.5
    after = InteractiveIsing.calculate(InteractiveIsing.H(), hterm, g)
    InteractiveIsing.graphstate(g)[2] = 0.25
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, proposal) ≈ after - before

    eps_fd = 1e-6
    old = InteractiveIsing.graphstate(g)[1]
    InteractiveIsing.graphstate(g)[1] = old + eps_fd
    e_plus = InteractiveIsing.calculate(InteractiveIsing.H(), hterm, g)
    InteractiveIsing.graphstate(g)[1] = old - eps_fd
    e_minus = InteractiveIsing.calculate(InteractiveIsing.H(), hterm, g)
    InteractiveIsing.graphstate(g)[1] = old
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, SingleSpinProposal(1, old, NoChange(), 1)) ≈
          (e_plus - e_minus) / (2eps_fd) atol = 1e-7

    half_turn = IsingGraph(
        continuous_layer,
        CosineInteraction(turns = 0.5);
        precision = Float64,
        adj = cosine_adj(1.0),
    )
    InteractiveIsing.graphstate(half_turn) .= [0.0, 1.0]
    @test InteractiveIsing.calculate(InteractiveIsing.H(), half_turn.hamiltonian, half_turn) ≈
          -cos(-π)

    discrete_a = Layer(1, Discrete(), StateSet(-1.0, 0.0, 1.0); periodic = false)
    discrete_b = Layer(1, Discrete(), StateSet(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0); periodic = false)
    mixed_clock = IsingGraph(discrete_a, discrete_b, CosineInteraction(); precision = Float64, adj = cosine_adj(1.0))
    InteractiveIsing.graphstate(mixed_clock) .= [-1.0, -2.0]
    @test InteractiveIsing.calculate(InteractiveIsing.H(), mixed_clock.hamiltonian, mixed_clock) ≈
          -cos(0 - 2π / 7)
    @test_throws ArgumentError InteractiveIsing.calculate(InteractiveIsing.d_iH(), mixed_clock.hamiltonian, mixed_clock, SingleSpinProposal(1, -1.0, NoChange(), 1))

    local_phase = IsingGraph(
        continuous_layer,
        CosineInteraction(phase = [0.0, π / 2]);
        precision = Float64,
        adj = cosine_adj(1.0),
    )
    InteractiveIsing.graphstate(local_phase) .= [0.0, 0.0]
    @test InteractiveIsing.calculate(InteractiveIsing.H(), local_phase.hamiltonian, local_phase) ≈
          -cos(π / 2) atol = 1e-15

    upper_phase = sparse([1], [2], [π / 4], 2, 2)
    upper = IsingGraph(
        continuous_layer,
        CosineInteraction(edge_phase = upper_phase, edge_phase_orientation = :upper);
        precision = Float64,
        adj = cosine_adj(1.0),
    )
    InteractiveIsing.graphstate(upper) .= [0.0, 0.25]
    @test InteractiveIsing.calculate(InteractiveIsing.H(), upper.hamiltonian, upper) ≈
          -cos(-π / 2 - π / 4)

    antisymmetric_phase = sparse([1, 2], [2, 1], [π / 4, -π / 4], 2, 2)
    antisymmetric = IsingGraph(
        continuous_layer,
        CosineInteraction(edge_phase = antisymmetric_phase, edge_phase_orientation = :antisymmetric);
        precision = Float64,
        adj = cosine_adj(1.0),
    )
    InteractiveIsing.graphstate(antisymmetric) .= [0.0, 0.25]
    @test InteractiveIsing.calculate(InteractiveIsing.H(), antisymmetric.hamiltonian, antisymmetric) ≈
          -cos(-π / 2 - π / 4)

    raw_phase = sparse([1], [2], [π / 4], 2, 2)
    raw = IsingGraph(
        continuous_layer,
        CosineInteraction(edge_phase = raw_phase, edge_phase_orientation = :raw);
        precision = Float64,
        adj = cosine_adj(1.0),
    )
    InteractiveIsing.graphstate(raw) .= [0.0, 0.25]
    @test InteractiveIsing.calculate(InteractiveIsing.H(), raw.hamiltonian, raw) ≈
          -0.5 * (cos(-π / 2 - π / 4) + cos(π / 2))

    invalid_phase = sparse([1, 2], [2, 1], [π / 4, π / 4], 2, 2)
    @test_throws ArgumentError IsingGraph(
        continuous_layer,
        CosineInteraction(edge_phase = invalid_phase, edge_phase_orientation = :antisymmetric);
        precision = Float64,
        adj = cosine_adj(1.0),
    )

    infinite_layer = Layer(2, Continuous(), StateSet(-Inf, Inf); periodic = false)
    @test_throws ArgumentError IsingGraph(
        infinite_layer,
        CosineInteraction();
        precision = Float64,
        adj = cosine_adj(1.0),
    )

    combo = IsingGraph(
        continuous_layer,
        Ising(c = ConstVal(0.0), b = 0) + CosineInteraction();
        precision = Float64,
        adj = cosine_adj(1.0),
    )
    @test InteractiveIsing.gethamiltonian(combo.hamiltonian, InteractiveIsing.CosineInteraction) isa
          InteractiveIsing.CosineInteraction
end

@testset "Hamiltonian Display" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)
    hts = g.hamiltonian

    hts_text = sprint(show, MIME"text/plain"(), hts)
    @test startswith(hts_text, "HamiltonianTerms")
    @test occursin("[1]: Quadratic", hts_text)
    @test occursin("└── parameters", hts_text)
    @test !occursin("Quadratic{Parameters", hts_text)

    hts_type_text = sprint(show, MIME"text/plain"(), typeof(hts))
    @test startswith(hts_type_text, "HamiltonianTerms")
    @test occursin("├── Quadratic", hts_type_text)
    @test occursin("├── Bilinear", hts_type_text)
    @test occursin("└── MagField", hts_type_text)
    @test occursin("c = Float32 [Defaulted]", hts_type_text)
    @test !occursin("Parameters{@NamedTuple", hts_type_text)

    term_type_text = sprint(show, MIME"text/plain"(), typeof(hts[1]))
    @test startswith(term_type_text, "Quadratic\n└── parameters")
    @test occursin("lp = OffsetArray [Defaulted]", term_type_text)
end

@testset "GaussianBernoulli GRBM Semantics" begin
    visible = Layer(2, StateSet(-Inf, Inf), Continuous(); periodic = false)
    hidden = Layer(2, StateSet(0, 1), Discrete(); periodic = false)
    w = Float64[
        0.2 -0.1
        0.3  0.4
    ]
    μ = Float64[0.1, -0.2]
    σ2 = Float64[0.7, 1.3]
    logσ2 = log.(σ2)
    b = Float64[0.05, -0.15]
    g = IsingGraph(visible, hidden, GaussianBernoulli(; w, μ, logσ2, b); precision = Float64)
    InteractiveIsing.graphstate(g) .= Float64[0.25, -0.5, 1, 0]

    hterm = g.hamiltonian
    v = Float64[0.25, -0.5]
    h = Float64[1, 0]
    logits = b .+ w' * (v ./ σ2)
    expected_joint = 0.5 * sum((v .- μ) .^ 2 ./ σ2) - dot(v ./ σ2, w * h) - dot(b, h)
    expected_marginal = 0.5 * sum((v .- μ) .^ 2 ./ σ2) - sum(log1p.(exp.(logits)))
    expected_joint_grad = (v .- μ .- w * h) ./ σ2
    expected_marginal_grad = (v .- μ .- w * (1 ./ (1 .+ exp.(-logits)))) ./ σ2

    @test InteractiveIsing.joint_energy(hterm, g) ≈ expected_joint
    @test InteractiveIsing.calculate(InteractiveIsing.H(), hterm, g) ≈ expected_joint
    @test InteractiveIsing.marginal_energy(hterm, g) ≈ expected_marginal
    @test InteractiveIsing.hidden_logits(hterm, g) ≈ logits
    @test InteractiveIsing.hidden_probabilities(hterm, g) ≈ 1 ./ (1 .+ exp.(-logits))
    @test InteractiveIsing.visible_energy_gradient(hterm, g) ≈ expected_joint_grad
    @test InteractiveIsing.marginal_visible_gradient(hterm, g) ≈ expected_marginal_grad

    visible_proposal = FlipProposal(1, v[1], 0.4, 1)
    expected_visible_ΔH = ((0.4 - μ[1])^2 - (v[1] - μ[1])^2) / (2 * σ2[1]) -
                          (0.4 - v[1]) * dot(w[1, :], h) / σ2[1]
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, visible_proposal) ≈ expected_visible_ΔH
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, SingleSpinProposal(1, v[1], NoChange(), 1)) ≈ expected_joint_grad[1]

    hidden_proposal = FlipProposal(4, 0.0, 1.0, 2)
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, hidden_proposal) ≈ -logits[2]
    @test_throws ArgumentError InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, SingleSpinProposal(4, 0.0, NoChange(), 2))

    eps_fd = 1e-6
    state = InteractiveIsing.graphstate(g)
    old = state[1]
    state[1] = old + eps_fd
    e_plus = InteractiveIsing.joint_energy(hterm, g)
    m_plus = InteractiveIsing.marginal_energy(hterm, g)
    state[1] = old - eps_fd
    e_minus = InteractiveIsing.joint_energy(hterm, g)
    m_minus = InteractiveIsing.marginal_energy(hterm, g)
    state[1] = old
    @test (e_plus - e_minus) / (2eps_fd) ≈ expected_joint_grad[1] atol = 1e-7
    @test (m_plus - m_minus) / (2eps_fd) ≈ expected_marginal_grad[1] atol = 1e-7

    deriv = InteractiveIsing.parameter_derivative(hterm, g)
    @test deriv.dw ≈ -v ./ σ2 .* h'
    @test deriv.dμ ≈ (μ .- v) ./ σ2
    expected_dlogσ2 = -0.5 .* (v .- μ) .^ 2 ./ σ2 .+ v .* (w * h) ./ σ2
    @test deriv.dlogσ2 ≈ expected_dlogσ2
    @test deriv.db ≈ -h

    sampler = GaussianBernoulliGibbsLangevin(; stepsize = 0.05, langevin_steps = 3, group_steps = 1, adjusted = false)
    context = InteractiveIsing.StatefulAlgorithms.init(sampler, (; model = g))
    update = InteractiveIsing.StatefulAlgorithms.step!(sampler, context)
    new_state = InteractiveIsing.graphstate(g)
    @test update.attempted == 1
    @test update.accepted == 1
    @test all(isfinite, new_state[InteractiveIsing.visible_indices(hterm, g)])
    @test all(x -> x == 0.0 || x == 1.0, new_state[InteractiveIsing.hidden_indices(hterm, g)])

    adjusted_sampler = GaussianBernoulliGibbsLangevin(; stepsize = 0.01, langevin_steps = 3, adjusted = true)
    adjusted_context = InteractiveIsing.StatefulAlgorithms.init(adjusted_sampler, (; model = g))
    adjusted_update = InteractiveIsing.StatefulAlgorithms.step!(adjusted_sampler, adjusted_context)
    @test adjusted_update.attempted == 1
    @test adjusted_update.accepted in (0, 1)
end

@testset "Hamiltonian Parameter Derivative Entry Point" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)
    spins = Float32[1, -2, 3, -4]
    InteractiveIsing.graphstate(g) .= spins

    hts = g.hamiltonian
    magfield = InteractiveIsing.gethamiltonian(hts, InteractiveIsing.MagField)

    @test InteractiveIsing.parameter_derivative(magfield, g) == (; db = -spins)

    deriv = InteractiveIsing.parameter_derivative(hts, g)
    @test deriv.db == -spins
    @test haskey(deriv, :dlp)
    @test haskey(deriv, :dJ)
end
