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
        Ising(c = ConstVal(0.0), b = 0.0) + CoulombHamiltonian(recalc = 1);
        periodic = (:x, :y),
        precision = Float64,
    )

    InteractiveIsing.graphstate(g) .= [1.0, -1.0, 0.25, 1.0, -0.5, 0.0, 1.0, -1.0]

    h = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
    InteractiveIsing.init!(h, g)

    spin_idx = 3
    analytic = InteractiveIsing.calculate(
        InteractiveIsing.d_iH(),
        h,
        g,
        SingleSpinProposal(spin_idx, InteractiveIsing.graphstate(g)[spin_idx], NoChange(), 1),
    )

    eps = 1e-6
    original = InteractiveIsing.graphstate(g)[spin_idx]

    InteractiveIsing.graphstate(g)[spin_idx] = original + eps
    h_plus = InteractiveIsing.instantiate(h, g)
    e_plus = 0.5 * sum(h_plus.ρ .* h_plus.u)

    InteractiveIsing.graphstate(g)[spin_idx] = original - eps
    h_minus = InteractiveIsing.instantiate(h, g)
    e_minus = 0.5 * sum(h_minus.ρ .* h_minus.u)

    InteractiveIsing.graphstate(g)[spin_idx] = original

    finite_difference = (e_plus - e_minus) / (2 * eps)
    @test analytic ≈ finite_difference atol = 1e-6 rtol = 1e-6
end

@testset "LayerTerm Coulomb scoping" begin
    layer_a = Layer(
        2,
        2,
        2,
        Continuous(),
        COULOMB_WG,
        LatticeConstants(1.0, 1.0, 1.0),
        StateSet(-1.5, 1.5);
        periodic = (:x, :y),
    )
    layer_b = Layer(
        2,
        2,
        2,
        Continuous(),
        COULOMB_WG,
        LatticeConstants(1.0, 1.0, 1.0),
        StateSet(-1.5, 1.5);
        periodic = (:x, :y),
    )
    hamiltonian = Ising(c = ConstVal(0.0), b = 0.0) + CoulombHamiltonian(layer = 2, recalc = 1)
    g = IsingGraph(layer_a, layer_b, hamiltonian; precision = Float64)

    layer_state = [1.0, -1.0, 0.25, 1.0, -0.5, 0.0, 1.0, -1.0]
    InteractiveIsing.graphstate(g)[InteractiveIsing.graphidxs(g[1])] .= -layer_state
    InteractiveIsing.graphstate(g)[InteractiveIsing.graphidxs(g[2])] .= layer_state

    h = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
    InteractiveIsing.init!(h, g)

    single = IsingGraph(
        2,
        2,
        2,
        Continuous(),
        COULOMB_WG,
        LatticeConstants(1.0, 1.0, 1.0),
        StateSet(-1.5, 1.5),
        Ising(c = ConstVal(0.0), b = 0.0) + CoulombHamiltonian(recalc = 1);
        periodic = (:x, :y),
        precision = Float64,
    )
    InteractiveIsing.graphstate(single) .= layer_state
    h_single = InteractiveIsing.gethamiltonian(single.hamiltonian, CoulombHamiltonian)
    InteractiveIsing.init!(h_single, single)

    outside = FlipProposal(3, InteractiveIsing.graphstate(g)[3], 0.9, 1)
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), h, g, outside) == 0.0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), h, g, SingleSpinProposal(3, InteractiveIsing.graphstate(g)[3], NoChange(), 1)) == 0.0

    rho_before = copy(h.ρ)
    accepted_outside = FlipProposal(3, InteractiveIsing.graphstate(g)[3], 0.9, 1, true)
    InteractiveIsing.update!(Metropolis(), h, g, accepted_outside)
    @test h.ρ == rho_before

    local_idx = 3
    global_idx = InteractiveIsing.graphidxs(g[2])[local_idx]
    global_prop = FlipProposal(global_idx, InteractiveIsing.graphstate(g)[global_idx], 0.9, 2)
    local_prop = FlipProposal(local_idx, layer_state[local_idx], 0.9, 1)

    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), h, g, global_prop) ≈
          InteractiveIsing.calculate(InteractiveIsing.ΔH(), h_single, single, local_prop)
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), h, g, SingleSpinProposal(global_idx, InteractiveIsing.graphstate(g)[global_idx], NoChange(), 2)) ≈
          InteractiveIsing.calculate(InteractiveIsing.d_iH(), h_single, single, SingleSpinProposal(local_idx, layer_state[local_idx], NoChange(), 1))
end

@testset "LayerTerm DepolField scoping" begin
    layer_a = Layer(2, 2, 2, Continuous(), StateSet(-1, 1); periodic = false)
    layer_b = Layer(2, 2, 2, Continuous(), StateSet(-1, 1); periodic = false)
    g = IsingGraph(layer_a, layer_b, Ising(c = ConstVal(0.0), b = 0.0) + DepolField(layer = 2); precision = Float64)
    InteractiveIsing.graphstate(g)[InteractiveIsing.graphidxs(g[1])] .= -1.0
    InteractiveIsing.graphstate(g)[InteractiveIsing.graphidxs(g[2])] .= 1.0

    h = InteractiveIsing.gethamiltonian(g.hamiltonian, DepolField)
    InteractiveIsing.init!(h, g)
    dpf_before = h.dpf[]
    m_before = h.M[]

    outside = FlipProposal(1, InteractiveIsing.graphstate(g)[1], 0.5, 1)
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), h, g, outside) == 0.0

    accepted_outside = FlipProposal(1, InteractiveIsing.graphstate(g)[1], 0.5, 1, true)
    InteractiveIsing.update!(Metropolis(), h, g, accepted_outside)
    @test h.dpf[] == dpf_before
    @test h.M[] == m_before
end
