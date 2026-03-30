using Test
using InteractiveIsing

const COULOMB_WG = @WG (;dr) -> dr == 1 ? 1.0 : 0.0 NN = 1

@testset "Coulomb d_iH" begin
    g = IsingGraph(
        2,
        2,
        2,
        Continuous(),
        COULOMB_WG,
        LatticeConstants(1.0, 1.0, 1.0),
        StateSet(-1.5, 1.5),
        Ising(c = ConstVal(0.0), b = StateLike(ConstFill, 0.0)) + CoulombHamiltonian(recalc = 1);
        periodic = (:x, :y),
        precision = Float64,
    )

    InteractiveIsing.graphstate(g) .= [1.0, -1.0, 0.25, 1.0, -0.5, 0.0, 1.0, -1.0]

    h = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
    InteractiveIsing.init!(h, g)

    spin_idx = 3
    analytic = InteractiveIsing.calculate(InteractiveIsing.d_iH(), h, g, spin_idx)

    eps = 1e-6
    original = InteractiveIsing.graphstate(g)[spin_idx]

    InteractiveIsing.graphstate(g)[spin_idx] = original + eps
    h_plus = InteractiveIsing.reconstruct(h, g)
    e_plus = 0.5 * sum(h_plus.ρ .* h_plus.u)

    InteractiveIsing.graphstate(g)[spin_idx] = original - eps
    h_minus = InteractiveIsing.reconstruct(h, g)
    e_minus = 0.5 * sum(h_minus.ρ .* h_minus.u)

    InteractiveIsing.graphstate(g)[spin_idx] = original

    finite_difference = (e_plus - e_minus) / (2 * eps)
    @test analytic ≈ finite_difference atol = 1e-6 rtol = 1e-6
end
