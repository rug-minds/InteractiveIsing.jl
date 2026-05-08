using Test
using InteractiveIsing
using LinearAlgebra
using SparseArrays

struct MissingFillArray{T,N} <: AbstractArray{T,N} end

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
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, 1) ≈ expected_joint_grad[1]

    hidden_proposal = FlipProposal(4, 0.0, 1.0, 2)
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, hidden_proposal) ≈ -logits[2]
    @test_throws ArgumentError InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, 4)

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
    context = InteractiveIsing.Processes.init(sampler, (; model = g))
    update = InteractiveIsing.Processes.step!(sampler, context)
    new_state = InteractiveIsing.graphstate(g)
    @test update.attempted == 1
    @test update.accepted == 1
    @test all(isfinite, new_state[InteractiveIsing.visible_indices(hterm, g)])
    @test all(x -> x == 0.0 || x == 1.0, new_state[InteractiveIsing.hidden_indices(hterm, g)])

    adjusted_sampler = GaussianBernoulliGibbsLangevin(; stepsize = 0.01, langevin_steps = 3, adjusted = true)
    adjusted_context = InteractiveIsing.Processes.init(adjusted_sampler, (; model = g))
    adjusted_update = InteractiveIsing.Processes.step!(adjusted_sampler, adjusted_context)
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
