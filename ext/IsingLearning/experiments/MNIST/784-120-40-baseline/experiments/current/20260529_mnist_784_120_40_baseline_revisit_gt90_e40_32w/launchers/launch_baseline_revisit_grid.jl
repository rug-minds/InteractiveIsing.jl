using Dates

const EXP_ROOT = normpath(joinpath(@__DIR__, ".."))
const BASELINE_ROOT = normpath(joinpath(EXP_ROOT, "..", "..", ".."))
const BASELINE_FILE = joinpath(BASELINE_ROOT, "mnist_784_120_40_adam.jl")
const SUMMARY_PATH = joinpath(EXP_ROOT, "grid_summary.csv")

delete!(ENV, "ISING_MNIST_IF_RESUME_FROM")
delete!(ENV, "ISING_MNIST_IF_RESUME_EPOCH")

include(BASELINE_FILE)

"""Append one baseline-grid summary row."""
function append_grid_row!(row::R) where {R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(SUMMARY_PATH) || filesize(SUMMARY_PATH) == 0
    open(SUMMARY_PATH, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return row
end

"""Return the last non-missing metric from a run row vector."""
function last_present(rows::R, field::Symbol) where {R<:AbstractVector}
    for idx in length(rows):-1:1
        value = getproperty(rows[idx], field)
        ismissing(value) || return value
    end
    return missing
end

"""Run one full baseline revisit configuration."""
function run_grid_point!(name::S, beta::T, lr::T, weight_decay::T) where {S<:AbstractString,T<:Real}
    outdir = joinpath(EXP_ROOT, name)
    config = InputFieldMNISTConfig(;
        workers = 32,
        epochs = 40,
        batchsize = 128,
        train_per_class = 5421,
        test_per_class = 892,
        train_eval_per_class = 100,
        eval_every = 5,
        hidden = 120,
        output_replicas = 4,
        sweeps = 500f0,
        β = FT(beta),
        lr = FT(lr),
        temp = 0.001f0,
        stepsize = 0.5f0,
        weight_scale = 0.005f0,
        weight_decay = FT(weight_decay),
        seed = 20260526,
        resume_from = "",
        resume_epoch = -1,
        outdir,
    )

    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] baseline revisit started name=", name)
    flush(stdout)
    started = time()
    try
        result = run_config!(config)
        append_grid_row!((;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            status = "ok",
            name,
            beta = Float64(beta),
            lr = Float64(lr),
            weight_decay = Float64(weight_decay),
            elapsed_seconds = round(time() - started; digits = 3),
            best_accuracy = result.best_accuracy,
            final_test_accuracy = last_present(result.rows, :test_accuracy),
            final_test_loss = last_present(result.rows, :test_loss),
            final_train_accuracy = last_present(result.rows, :train_accuracy),
            final_train_loss = last_present(result.rows, :train_loss),
            best_checkpoint = session_paths(outdir).best_path,
            latest_checkpoint = session_paths(outdir).latest_path,
            final_checkpoint = session_paths(outdir).final_path,
            outdir,
            error = "",
        ))
        println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] baseline revisit finished name=", name, " best_accuracy=", result.best_accuracy)
        flush(stdout)
        return result
    catch err
        mkpath(outdir)
        open(joinpath(outdir, "error.txt"), "w") do io
            showerror(io, err, catch_backtrace())
        end
        append_grid_row!((;
            timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            status = "error",
            name,
            beta = Float64(beta),
            lr = Float64(lr),
            weight_decay = Float64(weight_decay),
            elapsed_seconds = round(time() - started; digits = 3),
            best_accuracy = missing,
            final_test_accuracy = missing,
            final_test_loss = missing,
            final_train_accuracy = missing,
            final_train_loss = missing,
            best_checkpoint = "",
            latest_checkpoint = "",
            final_checkpoint = "",
            outdir,
            error = sprint(showerror, err),
        ))
        println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] baseline revisit failed name=", name, " error=", err)
        flush(stdout)
        return nothing
    end
end

"""Run the full baseline revisit grid sequentially."""
function main()
    mkpath(EXP_ROOT)
    configs = (
        ("beta5_lr0020_wd0_e40", 5.0f0, 0.0020f0, 0.0f0),
        ("beta5_lr0015_wd0_e40", 5.0f0, 0.0015f0, 0.0f0),
        ("beta3_lr0020_wd0_e40", 3.0f0, 0.0020f0, 0.0f0),
    )
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] baseline revisit launcher started root=", EXP_ROOT)
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] threads=", Threads.nthreads())
    flush(stdout)

    for (name, beta, lr, weight_decay) in configs
        run_grid_point!(name, beta, lr, weight_decay)
        GC.gc()
    end
    println("[", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "] baseline revisit launcher finished")
    flush(stdout)
    return nothing
end

main()
