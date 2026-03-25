using Test
using InteractiveIsing

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

@testset "Hamiltonian Parameter Derivative Entry Point" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)
    spins = Float32[1, -2, 3, -4]
    InteractiveIsing.setSpins!(g, spins)

    hts = g.hamiltonian
    magfield = InteractiveIsing.gethamiltonian(hts, InteractiveIsing.MagField)

    @test InteractiveIsing.parameter_derivative(magfield, g) == (; b = -spins)
    @test InteractiveIsing.parameter_derivative(hts, g) == (; b = -spins)
end
