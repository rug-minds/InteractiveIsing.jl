using Test

include("ManuscriptTools.jl")
using .ManuscriptTools

@testset "ManuscriptTools smoke tests" begin
    outdir = mktempdir()
    p = ManuscriptParams(;
        xL = 2,
        yL = 2,
        zL = 2,
        Steps_1 = 12,
        nrepeats = 3,
        outdir,
        save_figures = false,
        save_xlsx = false,
    )

    @test p.landau_mode == :independent
    @test Set(keys(landau_coefficients(p))) == Set((2, 4, 6, 8, 10))

    g = build_graph(p)
    @test length(graph_array(g)) == 8
    @test all(iszero, g.adj.diag)

    run = build_pulse_process(g, p)
    result = fetch_run(start_pulse!(g, run), run)
    @test !isempty(result.Pr)
    @test length(result.Pr) == length(result.voltage)
    @test length(result.H_total) == length(result.Pr)
    @test length(result.P_AFE_z) == length(result.Pr)

    saved = save_run_outputs(
        g,
        p;
        pulse = result,
        base_name = "smoke",
        save_figures = false,
        save_xlsx = true,
    )
    @test isfile(saved.csv_path_state)
    @test isfile(saved.csv_path_state_eu)
    @test isfile(saved.xlsx_path)

    disordered = update_params(
        p;
        apply_weak_landau_disorder = true,
        disorder_seed = 7,
    )
    coeffs_a = landau_coefficients(disordered)
    coeffs_b = landau_coefficients(disordered)
    @test coeffs_a[2] == coeffs_b[2]
    @test length(coeffs_a[2]) == 8
end

include(joinpath(@__DIR__, "..", "Manually checking", "ThreadedBasefile.jl"))

@testset "ThreadedBasefile smoke test" begin
    params = ManuscriptParams(;
        xL = 2,
        yL = 2,
        zL = 2,
        Steps_1 = 12,
        nrepeats = 3,
        outdir = mktempdir(),
        save_figures = false,
        save_xlsx = false,
    )
    jobs = [build_basefile_job(params; name = :threaded_smoke)]
    manager = run_threaded_basefile!(jobs; nworkers = 1, save_outputs = false)
    @test manager.state.names == Any[:threaded_smoke]
    @test !isnothing(manager.state.results[1])
    @test !isempty(manager.state.results[1].Pr)
    close(manager)
end
