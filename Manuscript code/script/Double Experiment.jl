include(joinpath(@__DIR__, "parallel_sweep_example2.jl"))

###############################################################################
# Multi-size experiment using the shared DSL helpers
###############################################################################

double_base = MT.ManuscriptParams(;
    outdir = raw"D:\Code\data\Manuscript\Double_experiment",
    xL = 20,
    yL = 20,
    zL = 10,
    JIsing = 1.0,
    Scale = 1.0,
    Screening = 0.5,
    Temp = 1.0f0,
    Temp_aneal = 3f0,
    time_fctr = 1.0,
    Steps_1 = 5000,
    Amp1 = 5.0,
    nrepeats = 2,
    proposal_delta = 0.1,
    algorithm_name = :metropolis,
    algorithm_kwargs = (;),
    landau_mode = :independent,
)

function size_paramsets(base)
    sizes = (
        (20, 20, 10),
        (20, 20, 5),
        (20, 20, 2),
    )

    return [
        MT.update_params(
            base;
            xL = dims[1],
            yL = dims[2],
            zL = dims[3],
            outdir = joinpath(base.outdir, "size=$(dims[1])x$(dims[2])x$(dims[3])"),
        )
        for dims in sizes
    ]
end

function double_experiment_paramsets(base)
    paramsets = MT.ManuscriptParams[]
    for p in size_paramsets(base)
        push!(paramsets, p)
        push!(paramsets, MT.update_params(
            p;
            algorithm_name = :local_langevin,
            algorithm_kwargs = (; stepsize = 0.05f0, adjusted = true),
            outdir = joinpath(p.outdir, "local_langevin"),
        ))
    end
    return paramsets
end

function run_double_experiment(; max_inflight = 3, capture = false)
    paramsets = double_experiment_paramsets(double_base)
    results = run_packaged_pulse_sweep_batched(paramsets; max_inflight, capture)

    for item in results
        println()
        println("Saved: ", item.paths.xlsx_path)
        println("Algorithm: ", item.params.algorithm_name)
        println("Size: ", (item.params.xL, item.params.yL, item.params.zL))
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_double_experiment(; max_inflight = 3, capture = false)
end
