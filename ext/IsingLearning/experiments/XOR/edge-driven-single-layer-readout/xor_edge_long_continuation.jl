include(joinpath(@__DIR__, "xor_edge_application_grid.jl"))

const EDGE_LONG_OUTDIR = joinpath(
    @__DIR__,
    "experiments",
    "current",
    "separate_input_output_lines_side16_nn9to10_seeds1to5_e10000",
)

"""Run the most robust wide-NN edge settings long enough to check convergence."""
function main()
    base = EdgeApplicationConfig(;
        epochs = 10_000,
        log_every = 500,
        snapshot_every = 0,
        outdir = EDGE_LONG_OUTDIR,
    )
    configs = edge_configs(base, collect(9:10), collect(1:5))
    return run_edge_grid!(base, configs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
