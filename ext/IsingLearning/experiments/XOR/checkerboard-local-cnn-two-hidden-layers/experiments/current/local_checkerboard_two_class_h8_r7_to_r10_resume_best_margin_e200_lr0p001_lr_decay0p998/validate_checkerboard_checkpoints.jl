using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

using Dates
using LuxCore
using Random
using Serialization

include(joinpath(@__DIR__, "xor_local_cnn_like_grid.jl"))

"""Parse a semicolon-separated list of checkerboard run directories."""
function validation_dirs(value::S) where {S<:AbstractString}
    return [strip(part) for part in split(value, ';') if !isempty(strip(part))]
end

"""Validate every requested config checkpoint with a higher repeat count."""
function validate_checkpoint_dir!(run_dir::P, repeats::Integer, checkpoint::S) where {P<:AbstractString,S<:AbstractString}
    rows = NamedTuple[]
    for config_dir in sort([path for path in readdir(run_dir; join = true) if isdir(path)])
        checkpoint_path = joinpath(config_dir, checkpoint)
        isfile(checkpoint_path) || continue
        payload = open(deserialize, checkpoint_path)
        config = copy_config(payload.config; eval_repeats = Int(repeats), plot = false)
        graph = cnn_xor_graph(config)
        layer = cnn_xor_layer(graph, config)
        st = LuxCore.initialstates(Random.MersenneTwister(config.seed + 21), layer)
        x, y = cnn_xor_dataset(config, FT)
        metrics = evaluate_cnn_xor(layer, payload.ps, st, x, y, config)
        row = (;
            run = basename(run_dir),
            config = basename(config_dir),
            checkpoint,
            repeats = Int(repeats),
            accuracy = metrics.accuracy,
            all_correct = metrics.all_correct,
            mse = metrics.mse,
            output_mse = metrics.output_mse,
            min_margin = metrics.min_margin,
            mean_margin = metrics.mean_margin,
            predictions = join(metrics.predictions, "|"),
            scores = join(round.(metrics.scores; digits = 5), "|"),
            margins = join(round.(metrics.margins; digits = 5), "|"),
        )
        push!(rows, row)
        println(row)
        flush(stdout)
    end
    out = joinpath(run_dir, "validation_$(repeats)_$(splitext(checkpoint)[1]).csv")
    isfile(out) && rm(out)
    for row in rows
        append_metrics!(out, row)
    end
    return out
end

"""Run high-repeat validation for checkerboard checkpoints selected by environment variables."""
function main()
    repeats = parse(Int, get(ENV, "ISING_XOR_VALIDATE_REPEATS", "1024"))
    checkpoint = get(ENV, "ISING_XOR_VALIDATE_CHECKPOINT", "best_margin_params.bin")
    dirs = validation_dirs(get(ENV, "ISING_XOR_VALIDATE_DIRS", ""))
    isempty(dirs) && throw(ArgumentError("set ISING_XOR_VALIDATE_DIRS to one or more run directories separated by semicolons"))
    outputs = [validate_checkpoint_dir!(dir, repeats, checkpoint) for dir in dirs]
    println("validated $(length(outputs)) run dirs at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    foreach(path -> println("  ", path), outputs)
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
