using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", "..", ".."))

using Dates
using Serialization

include(joinpath(@__DIR__, "mnist_local_paper_manager_grid.jl"))

"""Load serialized paper-manager parameters and rebuild a matching model."""
function load_paper_manager_checkpoint(
    path::P;
    test_per_class::I,
    free_reads::J,
    free_sweeps::K,
    outdir::S,
) where {P<:AbstractString,I<:Integer,J<:Integer,K<:Integer,S<:AbstractString}
    saved = open(path, "r") do io
        deserialize(io)
    end
    config = copy_config(
        saved.config;
        test_per_class = Int(test_per_class),
        free_reads = Int(free_reads),
        free_sweeps = Int(free_sweeps),
        outdir,
    )
    model = init_model(config)
    model.weights_0 .= saved.weights_0
    model.weights_12 .= saved.weights_12
    model.weights_2o .= saved.weights_2o
    model.weights_11 .= saved.weights_11
    model.weights_22 .= saved.weights_22
    model.weights_oo .= saved.weights_oo
    model.bias_1 .= saved.bias_1
    model.bias_2 .= saved.bias_2
    model.bias_o .= saved.bias_o
    sync_graph_couplings!(model)
    return model
end

"""Write a compact one-row CSV and markdown note for one checkpoint evaluation."""
function write_evaluation_outputs!(outdir::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    mkpath(outdir)
    append_row!(joinpath(outdir, "checkpoint_eval.csv"), row)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Paper-Manager Checkpoint Evaluation")
        println(io)
        println(io, "Use of this folder: larger held-out evaluation of one saved paper-manager checkpoint.")
        println(io)
        println(io, "- checkpoint: `$(row.checkpoint)`")
        println(io, "- test examples per class: `$(row.test_per_class)`")
        println(io, "- free reads / sweeps: `$(row.free_reads)` / `$(row.free_sweeps)`")
        println(io, "- accuracy: `$(row.accuracy)`")
        println(io, "- loss: `$(row.loss)`")
        println(io, "- prediction counts: `$(row.pred_counts)`")
    end
    return outdir
end

"""Evaluate one saved paper-manager checkpoint on a balanced MNIST test slice."""
function main()
    checkpoint = get(ENV, "ISING_MNIST_PM_EVAL_CHECKPOINT", "")
    isempty(checkpoint) && throw(ArgumentError("set ISING_MNIST_PM_EVAL_CHECKPOINT"))
    test_per_class = parse(Int, get(ENV, "ISING_MNIST_PM_EVAL_TEST_PER_CLASS", "100"))
    free_reads = parse(Int, get(ENV, "ISING_MNIST_PM_EVAL_FREE_READS", "3"))
    free_sweeps = parse(Int, get(ENV, "ISING_MNIST_PM_EVAL_FREE_SWEEPS", "50"))
    outdir = get(
        ENV,
        "ISING_MNIST_PM_EVAL_OUTDIR",
        @__DIR__,
    )

    model = load_paper_manager_checkpoint(
        checkpoint;
        test_per_class,
        free_reads,
        free_sweeps,
        outdir,
    )
    xtest, ytest = balanced_mnist(:test, test_per_class, model.config)
    seconds = @elapsed result = evaluate(model, xtest, ytest)
    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        checkpoint,
        test_per_class,
        free_reads,
        free_sweeps,
        seconds,
        accuracy = result.accuracy,
        loss = result.loss,
        pred_counts = join(result.pred_counts, "-"),
    )
    write_evaluation_outputs!(outdir, row)
    println(row)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
