# Complete example for the current common/ManuscriptTools.jl and
# Manually checking/ThreadedBasefile.jl APIs.
#
# Run from the repository root:
#
#   julia --threads=auto --project=. "Manuscript code/script/threaded_anneal_and_sweep_example.jl" anneal
#   julia --threads=auto --project=. "Manuscript code/script/threaded_anneal_and_sweep_example.jl" sweep
#   julia --threads=auto --project=. "Manuscript code/script/threaded_anneal_and_sweep_example.jl" all
#
# `anneal` runs one annealing experiment.
# `sweep` runs a threaded pulse sweep.
# `all` runs the annealing experiment first, then the threaded pulse sweep.

using Dates

include(joinpath(
    @__DIR__,
    "..",
    "Manually checking",
    "ThreadedBasefile_Ai.jl",
))

###############################################################################
# Output root
###############################################################################

const EXAMPLE_STAMP = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const EXAMPLE_OUTDIR = joinpath(
    @__DIR__,
    "..",
    "..",
    "runs",
    "threaded_anneal_and_sweep_" * EXAMPLE_STAMP,
)


const EXAMPLE_OUTDIR = joinpath(
    raw"D:\Code\data\Manuscript",
    "MyExperiment_" * EXAMPLE_STAMP,
)

###############################################################################
# Complete parameter block
###############################################################################

function full_example_params(; outdir = joinpath(EXAMPLE_OUTDIR, "base"))
    return MT.ManuscriptParams(;
        # Graph dimensions
        xL = 10,
        yL = 10,
        zL = 10,

        # Interaction and Coulomb parameters
        JIsing = 1.0,
        Scale = 1.0,
        Screening = 1.0,
        coulomb_recalc = 1000,

        # Temperature and run length
        Temp = 0.15f0,
        Temp_aneal = 2.0f0,
        time_fctr = 1.0,
        Steps_1 = 1200,

        # Pulse parameters
        Amp1 = 10.0,
        nrepeats = 3,

        # External and defect field scales
        linear_field_coeff = 1.0,
        defect_field_coeff = 0.0,

        # Proposal and dynamics
        proposal_delta = 0.1,
        algorithm_name = :local_langevin,
        algorithm_kwargs = (; stepsize = 0.02f0, adjusted = true),

        # Landau polynomial:
        # F(P) = a1*P^2 + b1*P^4 + c1*P^6 + d1*P^8 + e1*P^10
        a1 = -0.3,
        b1 = -2.1,
        c1 = 1.5,
        d1 = 0.0,
        e1 = 0.0,

        # Use `nothing` to construct coefficients from a1...e1.
        # Alternatively:
        # landau_coeffs = Dict(2 => -0.3, 4 => -2.1, 6 => 1.5)
        landau_coeffs = nothing,

        # Independent local-potential Landau coefficients. Do not put a1 in
        # adj(g)[1, 1] or any adjacency diagonal.
        landau_mode = :independent,

        # Optional per-site Landau disorder
        apply_weak_landau_disorder = false,
        coeff2_disorder_scale = 0.5,
        coeff4_disorder_scale = 1.0,
        coeff6_disorder_scale = 1.0,
        coeff8_disorder_scale = 0.2,
        coeff10_disorder_scale = 0.2,
        disorder_seed = 1234,

        # Logging and output
        log_diagnostics = true,
        capture = false,
        save_figures = true,
        save_xlsx = true,
        state_min = -1.5f0,
        state_max = 1.5f0,
        outdir,
    )
end

###############################################################################
# Example 1: one annealing experiment
###############################################################################

function run_anneal_example()
    params = full_example_params(;
        outdir = joinpath(EXAMPLE_OUTDIR, "anneal"),
    )
    job = build_basefile_job(
        params;
        name = :anneal_example,
        route = :anneal,
    )

    manager = run_threaded_basefile!(
        [job];
        nworkers = 1,
        save_outputs = true,
    )

    result = manager.state.results[1]
    paths = manager.state.paths[1]
    println("Anneal output: ", paths)
    return (; manager, params, result, paths)
end

###############################################################################
# Example 2: threaded pulse sweep
###############################################################################

function pulse_sweep_params()
    base = full_example_params()

    # Add or remove values here to change the sweep.
    scales = (0.5, 1.0)
    screenings = (0.5, 1.0)
    algorithms = (
        (
            label = "metropolis",
            name = :metropolis,
            kwargs = (;),
        ),
        (
            label = "local_langevin",
            name = :local_langevin,
            kwargs = (; stepsize = 0.02f0, adjusted = true),
        ),
    )

    params = MT.ManuscriptParams[]
    for Scale in scales
        for Screening in screenings
            for algorithm in algorithms
                label = join((
                    algorithm.label,
                    "Scale=$(Scale)",
                    "Screening=$(Screening)",
                ), "_")

                push!(params, MT.update_params(
                    base;
                    Scale,
                    Screening,
                    algorithm_name = algorithm.name,
                    algorithm_kwargs = algorithm.kwargs,
                    outdir = joinpath(EXAMPLE_OUTDIR, "pulse_sweep", label),
                ))
            end
        end
    end
    return params
end

function run_pulse_sweep_example(;
    nworkers = min(length(pulse_sweep_params()), Threads.nthreads()),
)
    paramsets = pulse_sweep_params()
    jobs = [
        build_basefile_job(
            params;
            name = Symbol(
                params.algorithm_name,
                "_Scale_",
                replace(string(params.Scale), "." => "_"),
                "_Screening_",
                replace(string(params.Screening), "." => "_"),
            ),
            route = :pulse,
        )
        for params in paramsets
    ]

    println(
        "Starting ",
        length(jobs),
        " pulse jobs with ",
        nworkers,
        " concurrent workers on ",
        Threads.nthreads(),
        " Julia threads.",
    )

    manager = run_threaded_basefile!(
        jobs;
        nworkers,
        save_outputs = true,
    )

    println("Pulse sweep outputs:")
    for (name, paths) in zip(manager.state.names, manager.state.paths)
        println("  ", name, ": ", paths)
    end
    return (; manager, paramsets, jobs)
end

###############################################################################
# Command-line entry point
###############################################################################

function main(args = ARGS)
    mode = isempty(args) ? "help" : lowercase(first(args))

    if mode == "anneal"
        return run_anneal_example()
    elseif mode == "sweep"
        return run_pulse_sweep_example()
    elseif mode == "all"
        anneal = run_anneal_example()
        close(anneal.manager)
        sweep = run_pulse_sweep_example()
        return (; anneal, sweep)
    elseif mode == "help"
        println("Usage: pass one mode: anneal, sweep, or all")
        return nothing
    else
        throw(ArgumentError("Unknown mode $(repr(mode)); use anneal, sweep, or all."))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
