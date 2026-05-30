using Dates

const MAIN_BATCH1_DIR = @__DIR__
const LEARNING_HELPER = normpath(joinpath(
    MAIN_BATCH1_DIR,
    "..",
    "20260530_182019_backend_update_contrastive_learning_retest",
    "local_langevin_learning_vs_process.jl",
))

include(LEARNING_HELPER)

"""Run the same one-sample full contrastive-learning benchmark used for backend isolation."""
function main_current_full_learning_once()
    mkpath(MAIN_BATCH1_DIR)
    println("marker: after include")
    flush(stdout)

    config = updated_config(langevin_learning_config(); batchsize = 1)
    println("marker: after config batchsize=$(config.batchsize)")
    flush(stdout)

    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
    println("marker: after data")
    flush(stdout)

    direct_setup = build_layer(config)
    println("marker: after direct build relaxation_steps=$(direct_setup.relaxation_steps)")
    flush(stdout)

    direct = time_direct_learning_minibatch!(direct_setup, xtrain, ytrain, config)
    println("direct_seconds_per_example=$(direct.seconds_per_example)")
    flush(stdout)

    process_setup = build_layer(config)
    println("marker: after process build")
    flush(stdout)

    serial_process = time_serial_process_learning_minibatch!(process_setup, xtrain, ytrain, config)
    ratio = serial_process.wall / direct.wall
    println("process_seconds_per_example=$(serial_process.seconds_per_example)")
    println("process_over_bespoke=$(ratio)")
    flush(stdout)

    result_path = joinpath(MAIN_BATCH1_DIR, "main_full_learning_batch1.csv")
    open(result_path, "w") do io
        println(io, "timestamp,threads,batchsize,direct_seconds_per_example,process_seconds_per_example,process_over_bespoke")
        println(
            io,
            join((
                Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                Threads.nthreads(),
                config.batchsize,
                direct.seconds_per_example,
                serial_process.seconds_per_example,
                ratio,
            ), ","),
        )
    end
    println("csv=$(result_path)")
    return (; direct, serial_process, ratio, result_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_current_full_learning_once()
end
