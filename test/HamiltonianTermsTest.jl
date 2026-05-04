using Test
using InteractiveIsing
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

@testset "GaussianBernoulli Template Semantics" begin
    g = IsingGraph(3, Continuous(); precision = Float32)
    spins = Float32[0.25, -0.5, 0.75]
    InteractiveIsing.graphstate(g) .= spins

    w = sparse(Float32[
        0.0  0.2 -0.1
        0.3  0.0  0.4
       -0.2  0.5  0.0
    ])
    self = Float32[1.1, 1.2, 1.3]
    σ = Float32[0.7, 0.8, 0.9]
    μ = Float32[0.05, -0.1, 0.2]
    b = Float32[0.3, -0.4, 0.5]

    hterm = GaussianBernoulli(g; w, self, σ, μ, b)
    proposal = FlipProposal(2, spins[2], Float32(0.125), 1)

    j = InteractiveIsing.at_idx(proposal)
    oldstate = spins[j]
    newstate = InteractiveIsing.to_val(proposal)
    cum = sum(w[i, j] * spins[i] for i in 1:length(spins))
    expected_ΔH = (newstate^2 - oldstate^2) * self[j] / σ[j]^2 +
                  (oldstate - newstate) * (cum + 2 * μ[j] * σ[j] + b[j]) +
                  Float32(0.5) * (newstate^2 - oldstate^2) +
                  (oldstate - newstate) * b[j]
    expected_d_iH = 2 * oldstate * self[j] / σ[j]^2 +
                    (cum + 2 * μ[j] * σ[j] + b[j]) +
                    oldstate + b[j]
    expected_H_i = oldstate^2 * self[j] / σ[j]^2 +
                   oldstate * (cum + 2 * μ[j] * σ[j] + b[j]) +
                   Float32(0.5) * oldstate^2 +
                   oldstate * b[j]

    expected_dw = zeros(Float32, size(w))
    for col in axes(w, 2)
        for ptr in nzrange(w, col)
            row = rowvals(w)[ptr]
            expected_dw[row, col] = spins[row] * spins[col]
        end
    end
    expected_dself = spins .^ 2 ./ σ .^ 2
    expected_dσ = @. -2 * spins^2 * self / σ^3 + 2 * μ * spins
    expected_dμ = @. 2 * σ * spins
    expected_db = 2 .* spins

    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), hterm, g, proposal) ≈ expected_ΔH
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), hterm, g, j) ≈ expected_d_iH
    @test InteractiveIsing.calculate(InteractiveIsing.H_i(), hterm, g, j) ≈ expected_H_i

    deriv = InteractiveIsing.parameter_derivative(hterm, g)
    @test Matrix(deriv.dw) ≈ expected_dw
    @test deriv.dself ≈ expected_dself
    @test deriv.dσ ≈ expected_dσ
    @test deriv.dμ ≈ expected_dμ
    @test deriv.db ≈ expected_db
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
