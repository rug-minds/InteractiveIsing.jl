using Test
using InteractiveIsing
using SparseArrays

@testset "Hamiltonian Term Field Access" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)
    hts = g.hamiltonian

    @test InteractiveIsing.getparam(hts, InteractiveIsing.Bilinear, Val(:adj)) ===
          InteractiveIsing.gethamiltonian(hts, InteractiveIsing.Bilinear).adj

    @test InteractiveIsing.getparam(hts, InteractiveIsing.MagField, Val(:b)) ===
          InteractiveIsing.gethamiltonian(hts, InteractiveIsing.MagField).b

    @test InteractiveIsing.getparam(hts, InteractiveIsing.Bilinear, :adj) ===
          InteractiveIsing.gethamiltonian(hts, InteractiveIsing.Bilinear).adj

    @test InteractiveIsing.getparam(hts, :adj) ===
          InteractiveIsing.getparam(hts, Val(:adj))

    duplicate_hts = InteractiveIsing.HamiltonianTerms(
        InteractiveIsing.Bilinear(g),
        InteractiveIsing.Bilinear(g),
    )

    @test_throws ErrorException InteractiveIsing.getparam(
        duplicate_hts,
        InteractiveIsing.Bilinear,
        Val(:adj),
    )

    @test InteractiveIsing.gethamiltonianfield(hts, InteractiveIsing.Bilinear, Val(:adj)) ===
          InteractiveIsing.getparam(hts, InteractiveIsing.Bilinear, Val(:adj))
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
    InteractiveIsing.setSpins!(g, spins)

    hts = g.hamiltonian
    magfield = InteractiveIsing.gethamiltonian(hts, InteractiveIsing.MagField)

    @test InteractiveIsing.parameter_derivative(magfield, g) == (; b = -spins)
    @test InteractiveIsing.parameter_derivative(hts, g) == (; b = -spins)
end
