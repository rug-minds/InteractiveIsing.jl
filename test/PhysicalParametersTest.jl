using Unitful
using SparseArrays

physical_test_weight(; dr) = 2u"eV"
physical_test_alltoall_weight(; dr, c1 = nothing, c2 = nothing, dc = nothing) = 2u"eV"

@testset "Physical Parameter Conversion" begin
    scales = PhysicalScales(
        energy = 1u"eV",
        temperature = 1u"K",
        length = 1u"nm",
        dipole = 1u"C*m",
    )

    g = IsingGraph(
        2,
        2,
        Continuous(),
        Ising(b = 2u"eV", c = 1);
        precision = Float64,
        physical_scales = scales,
        temperature = 300u"K",
    )
    mag = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.ExtField)
    @test temp(g) === 300.0
    @test mag.b[] === 2.0
    @test eltype(mag.b) === Float64

    setphysical!(g, InteractiveIsing.ExtField, :b, 3u"eV")
    @test mag.b[] === 3.0
    @test all(x -> x == 3.0u"eV", physicalvalue(g, InteractiveIsing.ExtField, :b))

    temp!(g, 125u"K")
    @test temp(g) === 125.0

    energy_temperature_scales = PhysicalScales(energy = 1u"meV")
    kbt_graph = IsingGraph(
        2,
        2,
        Continuous();
        precision = Float64,
        physical_scales = energy_temperature_scales,
        temperature = 2.5u"meV",
    )
    @test temp(kbt_graph) === 2.5
    temp!(kbt_graph, 3.5u"meV")
    @test temp(kbt_graph) === 3.5
end

@testset "Physical Parameter Errors" begin
    err = try
        IsingGraph(1, 1, Continuous(), ExtField(b = 1u"eV"); precision = Float64)
        nothing
    catch e
        e
    end

    @test err isa InteractiveIsing.MissingPhysicalScale
    @test occursin("energy", sprint(showerror, err))
    @test occursin("PhysicalScales", sprint(showerror, err))
end

@testset "Unitful Topology Lengths" begin
    top = SquareTopology((2, 2); lattice_constants = (2u"nm", 3u"nm"))
    @test InteractiveIsing.lattice_constants(top) == [2.0, 3.0]

    scaled = SquareTopology(
        (2, 2);
        lattice_constants = (2u"m", 300u"cm"),
        physical_scales = PhysicalScales(length = 1u"m"),
    )
    @test InteractiveIsing.lattice_constants(scaled) == [2.0, 3.0]

    lattice = LatticeTopology(
        (2.0u"m", 0.0u"m"),
        (0.0u"m", 300.0u"cm");
        physical_scales = PhysicalScales(length = 1u"m"),
    )
    @test InteractiveIsing.lattice_constants(lattice) == (2.0f0, 3.0f0)
end

@testset "Physical Weight Generator" begin
    scales = PhysicalScales(length = 1u"nm", energy = 1u"eV")
    wg = PhysicalWeightGenerator(WeightGenerator(physical_test_weight, 1), scales)
    g = IsingGraph(
        2,
        2,
        Continuous(),
        wg;
        periodic = false,
        precision = Float64,
        physical_scales = scales,
    )

    expected = [
        0.0 2.0 2.0 2.0
        2.0 0.0 2.0 2.0
        2.0 2.0 0.0 2.0
        2.0 2.0 2.0 0.0
    ]
    @test Matrix(adj(g)) == expected

    alltoall = PhysicalWeightGenerator(AllToAllWeightGenerator(physical_test_alltoall_weight), scales)
    alltoall_graph = IsingGraph(
        2,
        2,
        Continuous(),
        alltoall;
        periodic = false,
        precision = Float64,
        physical_scales = scales,
    )
    @test Matrix(adj(alltoall_graph)) == expected

    inferred_scales = PhysicalScales(energy = 1u"eV")
    inferred_wg = PhysicalWeightGenerator(WeightGenerator(physical_test_weight, 1))
    inferred_graph = IsingGraph(
        2,
        2,
        Continuous(),
        LatticeConstants(1u"nm", 1u"nm"),
        inferred_wg;
        periodic = false,
        precision = Float64,
        physical_scales = inferred_scales,
    )
    @test inferred_scales.length[] == 1u"nm"
    @test Matrix(adj(inferred_graph)) == expected
end

@testset "Coulomb Physical Parameters" begin
    scales = PhysicalScales(length = 1u"nm", dipole = 1u"C*m")
    g = IsingGraph(
        2,
        2,
        1,
        Continuous(),
        Ising(c = ConstVal(0.0), b = 0.0) +
        CoulombHamiltonian(scaling = 2u"C*m", screening = 3u"nm", recalc = 1);
        precision = Float64,
        physical_scales = scales,
    )
    h = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.CoulombHamiltonian)
    @test h.scaling[] === 2.0
    @test h.screen_top === 3.0
    @test h.screen_bot === 3.0
end
