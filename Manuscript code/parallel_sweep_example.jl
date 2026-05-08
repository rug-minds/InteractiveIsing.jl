# This file is a usage example for the helpers in `common/`.
# It does not edit or depend on Basefile.jl.

include(joinpath(@__DIR__, "common", "ManuscriptTools.jl"))
import InteractiveIsing
using InteractiveIsing.Processes
MT = ManuscriptTools

example_wg_skymion(; dc) = MT.weightfunc_skymion(; dc)
example_wg_shell_custom(; dc) = MT.weightfunc_shell(1.0, 1.0, 0.5, 1.0, 0.2, 0.1; dc)
function example_wg_inline(; dc)
    dx, dy, dz = dc
    r = sqrt(dx^2 + dy^2 + (2dz)^2)
    return 1 / r
end

###############################################################################
# 1. Basic single-run parameters
###############################################################################

base = MT.ManuscriptParams(;
    outdir = raw"D:\Code\data\Manuscript\Demo1_paralle2",
    xL = 40,
    yL = 40,
    zL = 10,
    JIsing = 0.5,
    Scale = 1.0,
    Screening = 0.01,
    Temp = 0.5f0,
    Temp_aneal = 3f0,
    time_fctr = 1.0,
    Steps_1 = 6000,
    Amp1 = 10.0,
    nrepeats = 2,
    algorithm_name = :metropolis,
    algorithm_kwargs = (;),
    a1 = -2.0,
    b1 = nothing,  # default: b1 = -(a1 + 3c1) / 2
    c1 = 10.0,
    landau_mode = :independent,
    landau_coeffs = nothing,
)

###############################################################################
# 2. Route design lives in the experiment file
###############################################################################

# `common/` supplies components, but this file owns the route design.
# For a new experiment, copy and edit these two functions here instead of
# changing `common/process_recipes.jl`.

function make_anneal_run(g, p)
    parts = MT.anneal_components(g, p)
    (; d, dynamics, annealing, M_Integrator, M_Logger, B_Logger, T_Logger) = parts
    point_repeat = max(1, round(Int, d.point_repeat))
    anneal_time = max(1, round(Int, d.anneal_time))

    Metro_T = @CompositeAlgorithm begin
        @alias dynamics = dynamics
        @alias M_Integrator = M_Integrator
        @alias M_Logger = M_Logger
        @alias B_Logger = B_Logger
        @alias T_Logger = T_Logger

        proposal = dynamics()
        total = M_Integrator(
            Δvalue = @transform(MT.accepted_proposal_delta, proposal),
        )
        @every point_repeat M_Logger(value = total)
        @every point_repeat B_Logger(
            value = @transform(x -> x.b[], dynamics.hamiltonian),
        )
        @every point_repeat T_Logger(
            value = @transform(InteractiveIsing.temp, dynamics.model),
        )
    end

    anneal_part = @CompositeAlgorithm begin
        @alias annealing = annealing
        @context metro_t = Metro_T()

        @every point_repeat annealing(model = metro_t.dynamics.model)
    end

    algorithm = @Routine begin
        @repeat anneal_time anneal_part()
    end
    return MT.anneal_run(algorithm, parts)
end

function make_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"))
    parts = MT.pulse_components(g, p; capture_dir)
    (; d, dynamics, pulse, M_Integrator, M_Logger, B_Logger, Graph_Logger) = parts
    point_repeat = max(1, round(Int, d.point_repeat))
    capture_interval1 = max(1, round(Int, d.capture_interval1))
    capture_interval2 = max(1, round(Int, d.capture_interval2))
    pulse_time = max(1, round(Int, d.pulse_time))
    relax_time = max(1, round(Int, d.relax_time))

    Metro_Pulse = @CompositeAlgorithm begin
        @alias dynamics = dynamics
        @alias M_Integrator = M_Integrator
        @alias M_Logger = M_Logger
        @alias B_Logger = B_Logger

        proposal = dynamics()
        total = M_Integrator(
            Δvalue = @transform(MT.accepted_proposal_delta, proposal),
        )
        @every point_repeat M_Logger(value = total)
        @every point_repeat B_Logger(
            value = @transform(x -> x.b[], dynamics.hamiltonian),
        )
    end

    pulse_part = @CompositeAlgorithm begin
        @alias pulse = pulse
        @alias Graph_Logger = Graph_Logger
        @context metro_pulse = Metro_Pulse()

        @every point_repeat pulse(
            hamiltonian = metro_pulse.dynamics.hamiltonian,
            M = metro_pulse.dynamics.M,
        )
        @every capture_interval1 Graph_Logger(
            array = @transform(MT.graph_array, metro_pulse.dynamics.model),
        )
    end

    relax_part = @CompositeAlgorithm begin
        @alias Graph_Logger = Graph_Logger
        @context metro_pulse = Metro_Pulse()

        @every capture_interval2 Graph_Logger(
            array = @transform(MT.graph_array, metro_pulse.dynamics.model),
        )
    end

    algorithm = @Routine begin
        @repeat pulse_time pulse_part()
        @repeat relax_time relax_part()
    end

    return MT.pulse_run(algorithm, parts)
end

###############################################################################
# 3. Example A: run one pulse simulation
###############################################################################

function example_single_pulse(base)
    g = MT.build_graph(base)
    pulse_run = make_pulse_run(g, base; capture_dir = joinpath(base.outdir, "single_pulse", "capture"))

    process = MT.start_pulse!(g, pulse_run; repeats = 1)
    pulse_result = MT.fetch_run(process, pulse_run)

    paths = MT.save_run_outputs(g, base; pulse = pulse_result)
    println("Single pulse saved to: ", paths.xlsx_path)
    return (; graph = g, run = pulse_run, result = pulse_result, paths)
end

###############################################################################
# 4. Example B: run one annealing simulation
###############################################################################

function example_single_anneal(base)
    g = MT.build_graph(base)
    anneal_run = make_anneal_run(g, base)

    process = MT.start_anneal!(g, anneal_run; repeats = 1)
    anneal_result = MT.fetch_run(process, anneal_run)

    paths = MT.save_run_outputs(g, base; anneal = anneal_result)
    println("Single anneal saved to: ", paths.xlsx_path)
    return (; graph = g, run = anneal_run, result = anneal_result, paths)
end

###############################################################################
# 5. Example C: sweep one parameter, start all processes first, fetch later
###############################################################################

function example_screening_sweep(base)
    screenings = (0.005, 0.1, 1, 5, 10)

    paramsets = [
        MT.update_params(
            base;
            Screening = screening,
            outdir = joinpath(base.outdir, "screening_sweep"),
        )
        for screening in screenings
    ]

    # Important pattern:
    # 1. start_pulse_sweep starts every process first.
    # 2. fetch_pulse_sweep waits only after all runs are already launched.
    runs = MT.start_pulse_sweep(paramsets; repeats = 1, run_builder = make_pulse_run)
    finished = MT.fetch_pulse_sweep(runs)

    for item in finished
        paths = MT.save_run_outputs(item.graph, item.params; pulse = item.result)
        println("Screening=$(item.params.Screening) saved to: ", paths.xlsx_path)
    end

    return finished
end

###############################################################################
# 6. Example D: sweep multiple parameters
###############################################################################

function example_2d_sweep(base)
    screenings = (0.01, 0.05)
    scales = (0.5, 1.0, 2.0)

    paramsets = [
        MT.update_params(
            base;
            Screening = screening,
            Scale = scale,
            outdir = joinpath(base.outdir, "scale_screening_sweep"),
        )
        for screening in screenings
        for scale in scales
    ]

    runs = MT.start_pulse_sweep(paramsets; repeats = 1, run_builder = make_pulse_run)
    finished = MT.fetch_pulse_sweep(runs)

    for item in finished
        paths = MT.save_run_outputs(item.graph, item.params; pulse = item.result)
        println("Scale=$(item.params.Scale), Screening=$(item.params.Screening) saved to: ", paths.xlsx_path)
    end

    return finished
end

###############################################################################
# 7. Example E: change the Landau polynomial
###############################################################################

function example_landau_variants(base)
    # Original Basefile coefficients:
    # E(P) = a1*P^2 + b1*P^4 + c1*P^6
    # Landau coefficients are passed directly as localpotential values.
    p1 = MT.update_params(
        base;
        landau_mode = :independent,
        a1 = -2.0,
        b1 = nothing,
        c1 = 10.0,
        outdir = joinpath(base.outdir, "landau_246"),
    )

    # Explicit coefficients. This is the clearer way when you already know
    # a, b, c and do not want b1 computed from a1/c1.
    p2 = MT.update_params(
        base;
        landau_mode = :independent,
        landau_coeffs = Dict(2 => -2.0, 4 => 14.0, 6 => 10.0),
        outdir = joinpath(base.outdir, "landau_explicit_246"),
    )

    # Higher-order Landau shape. PolynomialHamiltonian(order) supports arbitrary
    # even orders, so this can include 8th and 10th order terms.
    p3 = MT.update_params(
        base;
        landau_mode = :independent,
        landau_coeffs = Dict(2 => -6, 4 => 16.25, 6 => -16.73, 8 => 7.4, 10 => -1.2),
        outdir = joinpath(base.outdir, "landau_246810"),
    )

    # Independent localpotential style:
    #   Ising(localpotential=StateLike(...)) + Quartic(localpotential=StateLike(...)) ...
    # Then each term receives its own coefficient directly into lp[] and c[]=1.
    # This avoids coupling the higher-order terms to JIsing / adj(g).diag.
    p4 = MT.update_params(
        base;
        landau_mode = :independent,
        landau_coeffs = Dict(2 => -2.0, 4 => 14.0, 6 => 10.0, 8 => -1.0),
        outdir = joinpath(base.outdir, "landau_independent"),
    )

    # Per-site / per-dipole coefficients:
    # use landau_mode=:independent and landau_storage=Vector or OffsetArray.
    # The coefficient value can then be an array matching MT.graph_array(g).
    # Example template:
    #
    # coeff2 = fill(-2.0f0, base.xL, base.yL, base.zL)
    # coeff4 = fill(14.0f0, base.xL, base.yL, base.zL)
    # coeff6 = fill(10.0f0, base.xL, base.yL, base.zL)
    # coeff2[:, :, 1] .= -1.0f0
    # p_site = MT.update_params(
    #     base;
    #     landau_mode = :independent,
    #     landau_storage = Vector,
    #     landau_coeffs = Dict(2 => coeff2, 4 => coeff4, 6 => coeff6),
    # )

    # Pick one:
    p = p3
    # p = p1
    # p = p2
    # p = p4

    g = MT.build_graph(p)
    pulse_run = make_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"))

    process = MT.start_pulse!(g, pulse_run; repeats = 1)
    pulse_result = MT.fetch_run(process, pulse_run)

    paths = MT.save_run_outputs(g, p; pulse = pulse_result)
    println("Landau variant saved to: ", paths.xlsx_path)
    return (; graph = g, run = pulse_run, result = pulse_result, paths)
end

###############################################################################
# 8. Example F: change the weight function
###############################################################################

function example_custom_weight(base)
    # Option 1: use one of the predefined weight functions.
    wg_skymion = InteractiveIsing.@WG example_wg_skymion NN = 3

    # Option 2: use shell coupling with different physical constants.
    wg_shell_custom = InteractiveIsing.@WG example_wg_shell_custom NN = 3

    # Option 3: write an inline custom expression.
    wg_inline = InteractiveIsing.@WG example_wg_inline NN = 3

    # Pick one here:
    wg = wg_shell_custom
    # wg = wg_skymion
    # wg = wg_inline

    p = MT.update_params(
        base;
        outdir = joinpath(base.outdir, "custom_weight"),
        Screening = 0.01,
        Scale = 1.0,
    )

    g = MT.build_graph(p; wg)
    pulse_run = make_pulse_run(g, p; capture_dir = joinpath(p.outdir, "capture"))

    process = MT.start_pulse!(g, pulse_run; repeats = 1)
    pulse_result = MT.fetch_run(process, pulse_run)

    paths = MT.save_run_outputs(g, p; pulse = pulse_result)
    println("Custom weight run saved to: ", paths.xlsx_path)
    return (; graph = g, run = pulse_run, result = pulse_result, paths)
end

###############################################################################
# 9. Example G: run anneal first, then pulse on the same graph
###############################################################################

function example_anneal_then_pulse(base)
    g = MT.build_graph(base)

    anneal_run = make_anneal_run(g, base)
    anneal_process = MT.start_anneal!(g, anneal_run; repeats = 1)
    anneal_result = MT.fetch_run(anneal_process, anneal_run)

    pulse_run = make_pulse_run(g, base; capture_dir = joinpath(base.outdir, "anneal_then_pulse", "capture"))
    pulse_process = MT.start_pulse!(g, pulse_run; repeats = 1)
    pulse_result = MT.fetch_run(pulse_process, pulse_run)

    paths = MT.save_run_outputs(g, base; anneal = anneal_result, pulse = pulse_result)
    println("Anneal + pulse saved to: ", paths.xlsx_path)
    return (; graph = g, anneal_run, pulse_run, anneal_result, pulse_result, paths)
end

###############################################################################
# Choose what to run
###############################################################################

if abspath(PROGRAM_FILE) == @__FILE__
    # Start with exactly one uncommented line.
    # example_single_pulse(base)
    # example_single_anneal(base)
    # example_screening_sweep(base)
    # example_2d_sweep(base)
    example_landau_variants(base)
    # example_custom_weight(base)
    # example_anneal_then_pulse(base)
end
