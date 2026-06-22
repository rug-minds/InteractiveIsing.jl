using Test
using Random
using Unitful
using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

"""
    defect_hopping_graph(size, hamiltonian[, proposer]; periodic=false)

Build a small continuous graph for defect-hopping tests.
"""
function defect_hopping_graph(
    size::NTuple{N,Int},
    hamiltonian,
    proposer = nothing;
    periodic::Bool = false,
    physical_scales = nothing,
) where {N}
    layer = Layer(size..., Continuous(), StateSet(-10.0, 10.0); periodic)
    isnothing(proposer) && return IsingGraph(
        layer,
        hamiltonian;
        precision = Float64,
        initial_state = 0.0,
        physical_scales,
    )
    return IsingGraph(
        layer,
        hamiltonian,
        proposer;
        precision = Float64,
        initial_state = 0.0,
        physical_scales,
    )
end

defect_hopping_coulomb_weights(; dr) = dr == 1 ? 1.0 : 0.0

@testset "DefectHopping" begin
    @testset "Monte Carlo Model Interface" begin
        g = defect_hopping_graph((3,), Quadratic(c = ConstVal(0.0), localpotential = zeros(3)); periodic = true)
        neutral_hamiltonian = LocalPotentialShiftCoupling(2, 0.0)
        defects = DefectsModel(
            g;
            vacancies = MobileVacancies([1]; hamiltonian = neutral_hamiltonian),
            charges = MobileCharges([2]; hamiltonian = neutral_hamiltonian),
            electron_attempt_rate = 2.0,
        )

        @test g isa AbstractMonteCarloModel
        @test defects isa AddOnAbstractMonteCarloModel
        @test MobileChargeHopping === DefectsModel
        @test InteractiveIsing.requires(DefectsModel) == (AbstractIsingGraph,)
        @test InteractiveIsing.requires(defects) == (AbstractIsingGraph,)
        @test InteractiveIsing.dependson(defects) === g
    end

    @testset "Polynomial ΔH" begin
        g = defect_hopping_graph((2, 2), Quadratic(c = ConstVal(3.0), localpotential = zeros(4)))
        InteractiveIsing.graphstate(g) .= [1.0, 2.0, 3.0, 4.0]

        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 2.0, (1, 0), true, false)
        ΔE = InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, proposal)
        @test ΔE ≈ 3.0 * 2.0 * (2.0^2 - 1.0^2)

        invalid = InteractiveIsing.DefectHopProposal(1, 1, 1, 2.0, (-1, 0), false, false)
        @test isinf(InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, invalid))
    end

    @testset "Mixed Hamiltonian Contributions" begin
        hamiltonian = Ising(c = ConstVal(3.0), localpotential = zeros(4), b = zeros(4))
        g = defect_hopping_graph((2, 2), hamiltonian)
        InteractiveIsing.graphstate(g) .= [1.0, 2.0, 3.0, 4.0]

        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 2.0, (1, 0), true, false)
        ΔE = InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, proposal)
        @test ΔE ≈ 3.0 * 2.0 * (2.0^2 - 1.0^2)
    end

    @testset "Typed Defect Modes" begin
        linear_lp = zeros(2)
        field = zeros(2)
        hamiltonian =
            PolynomialHamiltonian(1; c = ConstVal(2.0), localpotential = linear_lp) +
            ExtField(b = field, c = 3.0)
        effects = (
            LocalPotentialShift(1, 0.25),
            ExtFieldShift(0.5),
        )
        proposer = DefectHopping(defects = [1], effects = effects)
        g = defect_hopping_graph((2,), hamiltonian, proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= [1.0, 4.0]

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 1.0, (1,), context.proposer.effects, true, false)

        @test context.proposer.effects isa Tuple{LocalPotentialShift{1,Float64,Float64},ExtFieldShift{Float64,Float64}}
        @test g.hamiltonian[1].lp == [0.25, 0.0]
        @test g.hamiltonian[2].b == [0.5, 0.0]
        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, g, proposal) ≈
            2.0 * 0.25 * (4.0 - 1.0) - 3.0 * 0.5 * (4.0 - 1.0)

        accepted = InteractiveIsing.accept(context.proposer, proposal)
        InteractiveIsing.update!(Metropolis(), context.hamiltonian, g, accepted)
        @test g.hamiltonian[1].lp == [0.0, 0.25]
        @test g.hamiltonian[2].b == [0.0, 0.5]
    end

    @testset "Hopping Energy Scale Is Independent From Local Mutation" begin
        defect_lp = zeros(2)
        effects = (LocalPotentialShift(2, 0.2; hopping_scale = 5.0),)
        proposer = DefectHopping(defects = [1], effects = effects)
        g = defect_hopping_graph((2,), Quadratic(c = ConstVal(1.0), localpotential = defect_lp), proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= [1.0, 3.0]

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 1.0, (1,), context.proposer.effects, true, false)

        @test g.hamiltonian.lp == [0.2, 0.0]
        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, g, proposal) ≈ 5.0 * 0.2 * (3.0^2 - 1.0^2)
    end

    @testset "Local Potential Scale Coupling" begin
        coupling = LocalPotentialScaleCoupling(2, 2.0)
        proposer = DefectHopping(defects = [1], hamiltonian = coupling)
        g = defect_hopping_graph((2,), Quadratic(c = ConstVal(2.0), localpotential = [2.0, 3.0]), proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= [2.0, 3.0]

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 1.0, (1,), context.proposer.effects, true, false)
        accepted = InteractiveIsing.accept(context.proposer, proposal)

        @test context.hamiltonian.lp == [4.0, 3.0]
        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, g, proposal) ≈ 38.0
        InteractiveIsing.update!(Metropolis(), context.hamiltonian, g, accepted)
        @test context.hamiltonian.lp == [2.0, 6.0]
    end

    @testset "ExtField Charge Coupling" begin
        effects = (ExtFieldChargeCoupling(),)
        proposer = DefectHopping(defects = [1], charge = 0.5, effects = effects)
        hamiltonian = Quadratic(c = ConstVal(0.0), localpotential = ConstFill(0.0)) + Bilinear() + ExtField(b = 2.0)
        g = defect_hopping_graph((2, 2, 2), hamiltonian, proposer)
        context = StatefulAlgorithms.init(Metropolis(), (; model = g))

        positive_z = InteractiveIsing.DefectHopProposal(1, 1, 5, 0.5, (0, 0, 1), context.proposer.effects, true, false)
        negative_z = InteractiveIsing.DefectHopProposal(1, 5, 1, 0.5, (0, 0, -1), context.proposer.effects, true, false)
        negative_charge = InteractiveIsing.DefectHopProposal(1, 1, 5, -0.5, (0, 0, 1), context.proposer.effects, true, false)
        invalid = InteractiveIsing.DefectHopProposal(1, 1, 1, 0.5, (0, 0, -1), context.proposer.effects, false, false)

        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, context.proposer, positive_z) ≈ -1.0
        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, context.proposer, negative_z) ≈ 1.0
        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, context.proposer, negative_charge) ≈ 1.0
        @test isinf(InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, context.proposer, invalid))
    end

    @testset "Physical Parameter Conversion" begin
        scales = PhysicalScales(energy = 1u"eV", charge = 1u"C")
        hamiltonian = ExtField(b = zeros(2), c = 3.0)
        effects = (
            ExtFieldShift(0.5u"eV"),
            ExtFieldChargeCoupling(),
        )
        proposer = DefectHopping(defects = [1], charge = 0.25u"C", effects = effects)
        g = defect_hopping_graph((2,), hamiltonian, proposer; periodic = true, physical_scales = scales)

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))

        @test context.proposer.charge ≈ 0.25
        @test context.proposer.effects isa Tuple{ExtFieldShift{Float64,Float64},ExtFieldChargeCoupling{Nothing,Float64}}
        @test g.hamiltonian.b == [0.5, 0.0]
    end

    @testset "Hopping Parameters Follow Graph State Type" begin
        wg = @WG defect_hopping_coulomb_weights NN = 1
        scales = PhysicalScales(energy = 1u"eV", charge = 1u"C")
        hamiltonian =
            Quadratic(c = ConstVal(0.0), localpotential = zeros(8)) +
            ExtField(b = zeros(8), c = 1.0) +
            CoulombHamiltonian(
                recalc = Inf,
                q_positive = 2.0u"C",
                q_negative = 1.0u"C",
                free_charge_split = 0.25,
            )
        g = IsingGraph(
            2,
            2,
            2,
            Continuous(),
            wg,
            LatticeConstants(1.0, 1.0, 1.0),
            StateSet(-1.5, 1.5),
            hamiltonian;
            periodic = (:x, :y),
            precision = Float32,
            initial_state = 0.0,
            physical_scales = scales,
        )
        charges = ChargeHopProposer(
            g;
            positive = [CartesianIndex(1, 1, 1)],
            negative = [CartesianIndex(2, 2, 1), CartesianIndex(2, 2, 2)],
            positive_effects = (
                CoulombChargeShift(2.0u"C"; split = 0.25),
                LocalPotentialShift(2, 0.125; hopping_scale = 3.0),
                ExtFieldShift(0.5u"eV"; hopping_scale = 4.0),
            ),
            negative_effects = (
                CoulombChargeShift(-1.0u"C"; split = 0.25),
                ExtFieldChargeCoupling(; hopping_scale = 5.0),
            ),
            positive_charge = 2.0u"C",
            negative_charge = -1.0u"C",
            electron_attempt_rate = 11.0,
        )
        T = eltype(InteractiveIsing.graphstate(g))
        coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)

        @test T === Float32
        @test eltype(charges) === T
        @test eltype(charges.positive) === T
        @test eltype(charges.negative) === T
        @test charges.positive.charge isa T
        @test charges.negative.charge isa T
        @test charges.positive_attempt_rate isa T
        @test charges.negative_attempt_rate isa T
        @test charges.positive.effects[1].charge isa T
        @test charges.positive.effects[1].split isa T
        @test charges.positive.effects[2].strength isa T
        @test charges.positive.effects[2].hopping_scale isa T
        @test charges.positive.effects[3].strength isa T
        @test charges.positive.effects[3].hopping_scale isa T
        @test charges.negative.effects[1].charge isa T
        @test charges.negative.effects[1].split isa T
        @test charges.negative.effects[2].hopping_scale isa T
        @test coulomb.q_positive isa T
        @test coulomb.q_negative isa T
        @test coulomb.free_charge_split isa T
    end

    @testset "Charge Hop Attempt Rates" begin
        g = defect_hopping_graph((3,), Quadratic(c = ConstVal(0.0), localpotential = zeros(3)); periodic = true)
        rng = MersenneTwister(7)
        neutral_effects = (LocalPotentialShift(2, 0.0),)

        negative_only = ChargeHopProposer(
            g;
            positive = [1],
            negative = [2],
            positive_effects = neutral_effects,
            negative_effects = neutral_effects,
            vacancy_attempt_rate = 0.0,
        )
        negative_only_proposer = InteractiveIsing.get_proposer(negative_only)
        for _ in 1:20
            @test rand(rng, negative_only_proposer).species isa NegativeFreeCharge
        end

        fast_negative = ChargeHopProposer(
            g;
            positive = [1],
            negative = [2],
            positive_effects = neutral_effects,
            negative_effects = neutral_effects,
            electron_attempt_rate = 100.0,
        )
        fast_negative_proposer = InteractiveIsing.get_proposer(fast_negative)
        negative_draws = count(_ -> rand(rng, fast_negative_proposer).species isa NegativeFreeCharge, 1:300)
        @test negative_draws > 285

        @test_throws ArgumentError ChargeHopProposer(
            g;
            positive = [1],
            negative = [2],
            positive_effects = neutral_effects,
            negative_effects = neutral_effects,
            vacancy_attempt_rate = -1.0,
        )
        @test_throws ArgumentError ChargeHopProposer(
            g;
            positive = [1],
            negative = [2],
            positive_effects = neutral_effects,
            negative_effects = neutral_effects,
            vacancy_attempt_rate = 1.0,
            electron_attempt_rate = 1.0,
        )
    end

    @testset "Type Stable Hot Paths" begin
        hamiltonian =
            Quadratic(c = ConstVal(1.0), localpotential = zeros(4)) +
            ExtField(b = zeros(4))
        g = defect_hopping_graph((4,), hamiltonian; periodic = true)
        charges = ChargeHopProposer(
            g;
            positive = [1],
            negative = [3],
            positive_effects = (LocalPotentialShift(2, 0.1), ExtFieldShift(0.2)),
            negative_effects = (LocalPotentialShift(2, -0.1),),
            positive_charge = 1.0,
            negative_charge = -1.0,
        )
        context = StatefulAlgorithms.init(Metropolis(), (; model = charges))
        proposal = InteractiveIsing.DefectHopProposal(
            1,
            1,
            2,
            charges.positive.charge,
            (1,),
            charges.positive.effects,
            true,
            false,
        )
        wrapped = InteractiveIsing.ChargeHopProposal(PositiveFreeCharge(), proposal)
        rng = MersenneTwister(11)

        @inferred rand(rng, charges.positive)
        @inferred InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, charges.positive, proposal)
        @inferred InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, charges, wrapped)
        @inferred InteractiveIsing.accept(charges.positive, proposal)
        @inferred InteractiveIsing.accept(charges, wrapped)
    end

    @testset "Double Well Background Is Not Defect Field" begin
        defect_lp = zeros(2)
        hamiltonian =
            Quadratic(c = ConstVal(-1.0), localpotential = ConstFill(1.0)) +
            Quartic(c = ConstVal(1.0), localpotential = ConstFill(1.0)) +
            Quadratic(c = ConstVal(1.0), localpotential = defect_lp)
        proposer = DefectHopping(defects = [1], charge = 1.0)
        g = defect_hopping_graph((2,), hamiltonian, proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= [1.0, 3.0]

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 1.0, (1,), true, false)

        @test g.hamiltonian[1].lp == ConstFill(1.0, 2)
        @test g.hamiltonian[2].lp == ConstFill(1.0, 2)
        @test g.hamiltonian[3].lp == [1.0, 0.0]
        @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), context.hamiltonian, g, proposal) ≈ 8.0
    end

    @testset "Accepted Hop Mutation" begin
        proposer = DefectHopping(defects = [1], charge = 2.0)
        g = defect_hopping_graph((2, 2), Quadratic(c = ConstVal(1.0), localpotential = zeros(4)), proposer)
        context = StatefulAlgorithms.init(Metropolis(), (; model = g))

        @test g.hamiltonian.lp == [2.0, 0.0, 0.0, 0.0]
        proposal = InteractiveIsing.DefectHopProposal(1, 1, 2, 2.0, (1, 0), true, false)
        accepted = InteractiveIsing.accept(context.proposer, proposal)
        InteractiveIsing.update!(Metropolis(), context.hamiltonian, g, accepted)

        @test InteractiveIsing.isaccepted(accepted)
        @test context.proposer.defect_idxs == [2]
        @test !context.proposer.occupancy[1]
        @test context.proposer.occupancy[2]
        @test g.hamiltonian.lp == [0.0, 2.0, 0.0, 0.0]
        @test InteractiveIsing.graphstate(g) == zeros(4)
        @test collect(InteractiveIsing.sampling_indices(g)) == collect(1:4)
    end

    @testset "Zero Temperature Uphill Rejection" begin
        proposer = DefectHopping(defects = [1], charge = 1.0)
        g = defect_hopping_graph((2,), Quadratic(c = ConstVal(1.0), localpotential = zeros(2)), proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= [1.0, 3.0]
        temp!(g, 0.0)

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        for _ in 1:10
            out = StatefulAlgorithms.step!(Metropolis(), context)
            @test out.ΔE ≈ 8.0
            @test !InteractiveIsing.isaccepted(out.proposal)
            @test context.proposer.defect_idxs == [1]
            @test g.hamiltonian.lp == [1.0, 0.0]
        end
    end

    @testset "Invalid Boundary Rejection" begin
        proposer = DefectHopping(defects = [1], charge = 1.0)
        g = defect_hopping_graph((1,), Quadratic(c = ConstVal(1.0), localpotential = zeros(1)), proposer; periodic = false)
        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        lp_before = copy(g.hamiltonian.lp)

        out = StatefulAlgorithms.step!(Metropolis(), context)

        @test !InteractiveIsing.isaccepted(out.proposal)
        @test isinf(out.ΔE)
        @test context.proposer.defect_idxs == [1]
        @test g.hamiltonian.lp == lp_before
    end

    @testset "Partial Periodicity Respects Nonperiodic Axis" begin
        nx, ny, nz = 4, 4, 3
        hamiltonian = Quadratic(c = ConstVal(1.0), localpotential = zeros(nx * ny * nz))
        g = IsingGraph(nx, ny, nz, Continuous(), StateSet(-10.0, 10.0), hamiltonian; periodic = (:x, :y), precision = Float64)
        layer = g[1]
        @test InteractiveIsing.whichperiodic(InteractiveIsing.topology(layer)) == (true, true, false)

        low_z = DefectHopping(g; defects = [CartesianIndex(2, 2, 1)], charge = 1.0)
        high_z = DefectHopping(g; defects = [CartesianIndex(2, 2, nz)], charge = 1.0)
        low_rng = MersenneTwister(1)
        high_rng = MersenneTwister(2)
        low_z_crossings = 0
        high_z_crossings = 0
        low_z_rejected = true
        high_z_rejected = true

        for _ in 1:2000
            proposal = rand(low_rng, low_z)
            if proposal.displacement == (0, 0, -1)
                low_z_crossings += 1
                low_z_rejected &= !proposal.valid && proposal.to_idx == proposal.from_idx
            end

            proposal = rand(high_rng, high_z)
            if proposal.displacement == (0, 0, 1)
                high_z_crossings += 1
                high_z_rejected &= !proposal.valid && proposal.to_idx == proposal.from_idx
            end
        end

        @test low_z_crossings > 0
        @test high_z_crossings > 0
        @test low_z_rejected
        @test high_z_rejected
    end

    @testset "Charge Conservation" begin
        proposer = DefectHopping(defects = [1, CartesianIndex(2, 2)], charge = 1.5)
        g = defect_hopping_graph((3, 3), Quadratic(c = ConstVal(1.0), localpotential = zeros(9)), proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= collect(1.0:9.0)
        temp!(g, 1.0e6)

        context = StatefulAlgorithms.init(Metropolis(), (; model = g))
        for _ in 1:50
            StatefulAlgorithms.step!(Metropolis(), context)
        end

        @test sum(g.hamiltonian.lp) ≈ 3.0
        @test count(context.proposer.occupancy) == 2
        @test length(unique(context.proposer.defect_idxs)) == 2
    end

    @testset "Neutral Coulomb Free-Charge Occupancy" begin
        wg = @WG defect_hopping_coulomb_weights NN = 1
        hamiltonian = Ising(c = ConstVal(0.0), b = zeros(8), localpotential = zeros(8)) +
            CoulombHamiltonian(recalc = 1, q_positive = 2.0, q_negative = 1.0, free_charge_split = 0.25)
        g = IsingGraph(
            2,
            2,
            2,
            Continuous(),
            wg,
            LatticeConstants(1.0, 1.0, 1.0),
            StateSet(-1.5, 1.5),
            hamiltonian;
            periodic = (:x, :y),
            precision = Float64,
            initial_state = 0.0,
        )
        vacancy_effects = (
            CoulombChargeShift(2.0; split = 0.25),
            LocalPotentialShift(2, 0.4),
            ExtFieldShift(0.3),
        )
        electron_effects = (CoulombChargeShift(-1.0; split = 0.25),)
        charges = ChargeHopProposer(
            g;
            positive = [CartesianIndex(1, 1, 1)],
            negative = [CartesianIndex(2, 2, 1), CartesianIndex(2, 2, 2)],
            positive_effects = vacancy_effects,
            negative_effects = electron_effects,
            positive_charge = 2.0,
            negative_charge = -1.0,
        )
        coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)

        @test state(charges).positive_idxs == charges.positive.defect_idxs
        @test state(charges).negative_idxs == charges.negative.defect_idxs
        @test InteractiveIsing.free_charge_total(coulomb) ≈ 0.0
        @test coulomb.positive_cell_occupancy[1, 1, 1] == 1
        @test coulomb.negative_cell_occupancy[2, 2, 1] == 1
        @test coulomb.negative_cell_occupancy[2, 2, 2] == 1
        @test coulomb.ρ[1, 1, 1] ≈ 1.5
        @test coulomb.ρ[1, 1, 2] ≈ 0.5
        @test coulomb.ρ[2, 2, 1] ≈ -0.75
        @test coulomb.ρ[2, 2, 2] ≈ -1.0
        @test coulomb.ρ[2, 2, 3] ≈ -0.25
        @test sum(coulomb.ρ) ≈ 0.0 atol = 1.0e-12
        @test g.hamiltonian[1].lp[1] ≈ 0.4
        @test g.hamiltonian[3].b[1] ≈ 0.3

        context = StatefulAlgorithms.init(Metropolis(), (; model = charges))
        coulomb = InteractiveIsing.gethamiltonian(context.hamiltonian, CoulombHamiltonian)
        @test InteractiveIsing.free_charge_total(coulomb) ≈ 0.0
        @test coulomb.ρ[1, 1, 1] ≈ 1.5
        @test coulomb.ρ[1, 1, 2] ≈ 0.5

        proposal = InteractiveIsing.DefectHopProposal(
            1,
            1,
            2,
            1.0,
            (1, 0, 0),
            charges.positive.effects,
            true,
            false,
        )
        wrapped_proposal = InteractiveIsing.ChargeHopProposal(PositiveFreeCharge(), proposal)
        coulomb.recalc_tracker[] = 1
        ρ_before = copy(coulomb.ρ)
        ρhat_before = copy(coulomb.ρhat)
        uhat_before = copy(coulomb.uhat)
        u_before = copy(coulomb.u)
        pos_before = copy(coulomb.positive_cell_occupancy)
        ΔE = InteractiveIsing.calculate(InteractiveIsing.ΔH(), coulomb, charges, wrapped_proposal)
        @test coulomb.recalc_tracker[] == 1
        @test coulomb.ρ == ρ_before
        @test coulomb.ρhat == ρhat_before
        @test coulomb.uhat == uhat_before
        @test coulomb.u == u_before
        @test coulomb.positive_cell_occupancy == pos_before

        InteractiveIsing.recalc!(coulomb)
        energy_before = 0.5 * sum(coulomb.ρ .* coulomb.u)
        InteractiveIsing.move_cell_free_charge!(coulomb, PositiveFreeCharge(), CartesianIndex(1, 1, 1), CartesianIndex(2, 1, 1))
        InteractiveIsing.rebuild_charge_density!(coulomb, g[1])
        InteractiveIsing.recalc!(coulomb)
        energy_after = 0.5 * sum(coulomb.ρ .* coulomb.u)
        coulomb.ρ .= ρ_before
        coulomb.ρhat .= ρhat_before
        coulomb.uhat .= uhat_before
        coulomb.u .= u_before
        coulomb.positive_cell_occupancy .= pos_before

        @test ΔE ≈ energy_after - energy_before

        accepted = InteractiveIsing.accept(context.proposer, wrapped_proposal)
        InteractiveIsing.update!(Metropolis(), context.hamiltonian, charges, accepted)
        @test coulomb.positive_cell_occupancy[1, 1, 1] == 0
        @test coulomb.positive_cell_occupancy[2, 1, 1] == 1
        @test coulomb.ρ[1, 1, 1] ≈ 0.0 atol = 1.0e-12
        @test coulomb.ρ[1, 1, 2] ≈ 0.0 atol = 1.0e-12
        @test coulomb.ρ[2, 1, 1] ≈ 1.5
        @test coulomb.ρ[2, 1, 2] ≈ 0.5
        @test sum(coulomb.ρ) ≈ 0.0 atol = 1.0e-12
        local_ρ = copy(coulomb.ρ)
        local_u = copy(coulomb.u)
        InteractiveIsing.rebuild_charge_density!(coulomb, g[1])
        InteractiveIsing.recalc!(coulomb)
        @test coulomb.ρ ≈ local_ρ
        @test coulomb.u ≈ local_u

        electron_proposal = InteractiveIsing.DefectHopProposal(
            1,
            4,
            3,
            1.0,
            (-1, 0, 0),
            charges.negative.effects,
            true,
            false,
        )
        wrapped_electron = InteractiveIsing.ChargeHopProposal(NegativeFreeCharge(), electron_proposal)
        lp_before = copy(g.hamiltonian[1].lp)
        b_before = copy(g.hamiltonian[3].b)
        accepted_electron = InteractiveIsing.accept(context.proposer, wrapped_electron)
        InteractiveIsing.update!(Metropolis(), context.hamiltonian, charges, accepted_electron)

        @test coulomb.negative_cell_occupancy[2, 2, 1] == 0
        @test coulomb.negative_cell_occupancy[1, 2, 1] == 1
        @test g.hamiltonian[1].lp == lp_before
        @test g.hamiltonian[3].b == b_before
        @test InteractiveIsing.free_charge_total(coulomb) ≈ 0.0
    end

    @testset "Physical Coulomb Charge Conversion" begin
        wg = @WG defect_hopping_coulomb_weights NN = 1
        scales = PhysicalScales(charge = 1u"C")
        hamiltonian = Ising(c = ConstVal(0.0), b = zeros(8), localpotential = zeros(8)) +
            CoulombHamiltonian(recalc = Inf, q_positive = 0.8u"C", q_negative = 0.4u"C", free_charge_split = 0.25)
        g = IsingGraph(
            2,
            2,
            2,
            Continuous(),
            wg,
            LatticeConstants(1.0, 1.0, 1.0),
            StateSet(-1.5, 1.5),
            hamiltonian;
            periodic = (:x, :y),
            precision = Float64,
            initial_state = 0.0,
            physical_scales = scales,
        )
        charges = ChargeHopProposer(
            g;
            positive = [CartesianIndex(1, 1, 1)],
            negative = [CartesianIndex(2, 2, 1), CartesianIndex(2, 2, 2)],
            positive_effects = (CoulombChargeShift(0.8u"C"; split = 0.25),),
            negative_effects = (CoulombChargeShift(-0.4u"C"; split = 0.25),),
            positive_charge = 0.8u"C",
            negative_charge = -0.4u"C",
        )
        coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)

        @test charges.positive.charge ≈ 0.8
        @test charges.negative.charge ≈ -0.4
        @test charges.positive.effects[1].charge ≈ 0.8
        @test charges.negative.effects[1].charge ≈ 0.4
        @test coulomb.q_positive ≈ 0.8
        @test coulomb.q_negative ≈ 0.4
        @test InteractiveIsing.free_charge_total(coulomb) ≈ 0.0
    end

    @testset "Coulomb Recalc Schedule Controls" begin
        wg = @WG defect_hopping_coulomb_weights NN = 1
        hamiltonian = Ising(c = ConstVal(0.0), b = zeros(8), localpotential = zeros(8)) +
            CoulombHamiltonian(recalc = Inf, q_positive = 2.0, q_negative = 1.0, free_charge_split = 0.5)
        g = IsingGraph(
            2,
            2,
            2,
            Continuous(),
            wg,
            LatticeConstants(1.0, 1.0, 1.0),
            StateSet(-1.5, 1.5),
            hamiltonian;
            periodic = (:x, :y),
            precision = Float64,
            initial_state = 0.0,
        )
        coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
        @test isinf(coulomb.recalc_steps)
        @test coulomb.recalc_tracker[] == 1

        spin_proposal = FlipProposal(1, 0.0, 0.5, 1, true)
        u_before_spin = copy(coulomb.u)
        InteractiveIsing.update!(Metropolis(), coulomb, g, spin_proposal)
        @test coulomb.u == u_before_spin

        charges = ChargeHopProposer(
            g;
            positive = [CartesianIndex(1, 1, 1)],
            negative = [CartesianIndex(2, 2, 1), CartesianIndex(2, 2, 2)],
            positive_effects = (CoulombChargeShift(2.0; split = 0.5),),
            negative_effects = (CoulombChargeShift(-1.0; split = 0.5),),
            positive_charge = 2.0,
            negative_charge = -1.0,
        )
        context = StatefulAlgorithms.init(Metropolis(), (; model = charges))
        coulomb = InteractiveIsing.gethamiltonian(context.hamiltonian, CoulombHamiltonian)
        u_before_defect = copy(coulomb.u)
        defect_proposal = InteractiveIsing.DefectHopProposal(
            1,
            charges.positive.defect_idxs[1],
            charges.positive.defect_idxs[1] + 1,
            charges.positive.charge,
            (1, 0, 0),
            charges.positive.effects,
            true,
            false,
        )
        accepted_defect = InteractiveIsing.ChargeHopProposal(
            PositiveFreeCharge(),
            InteractiveIsing.accept(charges.positive, defect_proposal),
        )
        InteractiveIsing.update!(Metropolis(), context.hamiltonian, charges, accepted_defect)
        @test coulomb.u != u_before_defect
        @test coulomb.recalc_tracker[] == 1

        offset_hamiltonian = Ising(c = ConstVal(0.0), b = zeros(8), localpotential = zeros(8)) +
            CoulombHamiltonian(recalc = 5, recalc_offset = 2, q_positive = 2.0, q_negative = 1.0)
        offset_graph = IsingGraph(
            2,
            2,
            2,
            Continuous(),
            wg,
            LatticeConstants(1.0, 1.0, 1.0),
            StateSet(-1.5, 1.5),
            offset_hamiltonian;
            periodic = (:x, :y),
            precision = Float64,
            initial_state = 0.0,
        )
        offset_coulomb = InteractiveIsing.gethamiltonian(offset_graph.hamiltonian, CoulombHamiltonian)
        @test offset_coulomb.recalc_steps == 5
        @test offset_coulomb.recalc_tracker[] == 3
    end

    @testset "Coulomb Free-Charge Neutrality Validation" begin
        wg = @WG defect_hopping_coulomb_weights NN = 1
        hamiltonian = Ising(c = ConstVal(0.0), b = 0.0) +
            CoulombHamiltonian(recalc = 1, q_positive = 2.0, q_negative = 1.0, free_charge_split = 0.5)
        g = IsingGraph(
            2,
            2,
            2,
            Continuous(),
            wg,
            LatticeConstants(1.0, 1.0, 1.0),
            StateSet(-1.5, 1.5),
            hamiltonian;
            periodic = (:x, :y),
            precision = Float64,
            initial_state = 0.0,
        )
        coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
        InteractiveIsing.add_cell_free_charge!(coulomb, PositiveFreeCharge(), CartesianIndex(1, 1, 1))
        InteractiveIsing.add_cell_free_charge!(coulomb, NegativeFreeCharge(), CartesianIndex(2, 1, 1))

        @test InteractiveIsing.free_charge_total(coulomb) ≈ 1.0
        @test_throws ArgumentError InteractiveIsing.rebuild_charge_density!(coulomb, g[1])

        InteractiveIsing.add_cell_free_charge!(coulomb, NegativeFreeCharge(), CartesianIndex(2, 2, 1))
        @test InteractiveIsing.free_charge_total(coulomb) ≈ 0.0
        InteractiveIsing.rebuild_charge_density!(coulomb, g[1])
        @test sum(coulomb.ρ) ≈ 0.0 atol = 1.0e-12
    end

    @testset "Composite With LocalLangevin" begin
        proposer = DefectHopping(defects = [1], charge = 0.5)
        g = defect_hopping_graph((2, 2), Quadratic(c = ConstVal(1.0), localpotential = zeros(4)), proposer; periodic = true)
        InteractiveIsing.graphstate(g) .= 0.25
        temp!(g, 1.0)

        langevin_algorithm = LocalLangevin(stepsize = 0.01, adjusted = false)
        defect_algorithm = Metropolis()
        algorithm = @CompositeAlgorithm begin
            @alias langevin = langevin_algorithm
            @alias defect_metro = defect_algorithm

            langevin()
            @every 2 defect_metro()
        end
        process = InlineProcess(
            algorithm,
            StatefulAlgorithms.Init(:langevin; model = g),
            StatefulAlgorithms.Init(:defect_metro; model = g);
            repeats = 8,
        )
        run(process)

        @test all(isfinite, InteractiveIsing.graphstate(g))
        @test sum(g.hamiltonian.lp) ≈ 0.5
        @test collect(InteractiveIsing.sampling_indices(g)) == collect(1:4)
    end

    @testset "Immutable Local Potential Rejection" begin
        proposer = DefectHopping(defects = [1])
        g = defect_hopping_graph((2, 2), Quadratic(c = ConstVal(1.0), localpotential = ConstFill(0.0)), proposer)
        @test_throws ArgumentError StatefulAlgorithms.init(Metropolis(), (; model = g))
    end
end
