using Dates

const PROCESS_HANG_BUILDUP_DIR = @__DIR__
const PROCESS_HANG_BUILDUP_BASELINE = normpath(joinpath(@__DIR__, "..", "..", "..", "mnist_784_120_40_adam.jl"))
include(PROCESS_HANG_BUILDUP_BASELINE)

const PROCESS_HANG_BUILDUP_TIMEOUT_SECONDS = 15.0
const PROCESS_HANG_BUILDUP_ENTRYPOINTS = (:process_run, :process_inline, :inline_process)
const PROCESS_HANG_BUILDUP_STAGE_ORDER = (
    :free_phase_short,
    :free_phase_full,
    :free_then_nudged,
    :contrastive_static_beta,
    :contrastive_runtime_beta,
)

"""Return the results CSV path for the staged process hang diagnostic."""
function buildup_results_path()
    return joinpath(PROCESS_HANG_BUILDUP_DIR, "process_hang_buildup.csv")
end

"""Return the per-case scratch file used by one child subprocess."""
function buildup_case_result_path(stage::S, entrypoint::E) where {S,E}
    return joinpath(PROCESS_HANG_BUILDUP_DIR, "case_$(stage)_$(entrypoint).txt")
end

"""Return the stdout log path for one child subprocess."""
function buildup_stdout_path(stage::S, entrypoint::E) where {S,E}
    return joinpath(PROCESS_HANG_BUILDUP_DIR, "case_$(stage)_$(entrypoint).stdout.txt")
end

"""Return the stderr log path for one child subprocess."""
function buildup_stderr_path(stage::S, entrypoint::E) where {S,E}
    return joinpath(PROCESS_HANG_BUILDUP_DIR, "case_$(stage)_$(entrypoint).stderr.txt")
end

"""Return the single-sample config used by the staged hang diagnosis."""
function buildup_config()
    return InputFieldMNISTConfig(;
        workers = 1,
        epochs = 1,
        batchsize = 1,
        scheduler = "spawn",
        chunk_size = 0,
        train_per_class = 80,
        test_per_class = 1,
        train_eval_per_class = 0,
        eval_every = 1,
        sweeps = 500f0,
        β = 5f0,
        lr = 0.0015f0,
        weight_decay = 0f0,
        temp = 0.001f0,
        stepsize = 0.5f0,
        seed = 20260526,
        outdir = String(PROCESS_HANG_BUILDUP_DIR),
    )
end

"""Build a short free-phase routine to separate loop-structure bugs from long runtimes."""
function free_phase_short_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    dynamics_algorithm = deepcopy(layer.dynamics_algorithm)
    short_steps = min(layer.free_relaxation_steps, 16)
    n_units = layer.nunits

    return StatefulAlgorithms.@Routine begin
        @state x
        @state input_hidden_w
        @state input_pattern = zeros(FT, n_units)
        @state equilibrium_state = zeros(n_units)
        @alias dynamics = dynamics_algorithm

        # Keep the exact free-phase shape, but with a short repeat count.
        II.resetstate!(dynamics.model)
        ApplyProjectedInputFieldRef!(dynamics.model, input_hidden_w, x, input_pattern)
        model = @repeat short_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(graph -> II.state(graph), model))
    end
end

"""Build a composite that runs the real free and nudged phases without the gradient step."""
function free_then_nudged_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    free_phase = input_field_free_phase_algorithm(layer)
    nudged_phase = input_field_nudged_phase_algorithm(layer)

    return StatefulAlgorithms.@CompositeAlgorithm begin
        @state x
        @state y
        @state input_hidden_w

        # This isolates the two-phase context wiring before adding gradient accumulation.
        @context free_context = free_phase()
        @context nudged_context = nudged_phase()
        @bind x => free_context.x
        @bind x => nudged_context.x
        @bind y => nudged_context.y
        @bind input_hidden_w => free_context.input_hidden_w
        @bind input_hidden_w => nudged_context.input_hidden_w
        @merge free_context.input_pattern, nudged_context.input_pattern
        @merge free_context.equilibrium_state, nudged_context.equilibrium_state
    end
end

"""Build the full contrastive composite with static beta state instead of runtime input."""
function contrastive_static_beta_algorithm(layer::L) where {L<:IsingLearning.LayeredIsingGraphLayer}
    default_β = layer.β
    free_phase = input_field_free_phase_algorithm(layer)
    nudged_phase = input_field_nudged_phase_algorithm(layer)

    return StatefulAlgorithms.@CompositeAlgorithm begin
        @state x
        @state y
        @state buffers
        @state input_hidden_w
        @state phase_beta = default_β

        # This preserves the full dataflow while removing the runtime-input path.
        @context free_context = free_phase()
        @context nudged_context = nudged_phase()
        @bind x => free_context.x
        @bind x => nudged_context.x
        @bind y => nudged_context.y
        @bind input_hidden_w => free_context.input_hidden_w
        @bind input_hidden_w => nudged_context.input_hidden_w
        @merge free_context.input_pattern, nudged_context.input_pattern
        @merge free_context.equilibrium_state, nudged_context.equilibrium_state
        @bind phase_beta => nudged_context.phase_beta

        AccumulateInputFieldGradientRef!(
            nudged_context.dynamics.model,
            nudged_context.nudged_state,
            free_context.equilibrium_state,
            x,
            buffers,
            phase_beta,
        )
    end
end

"""Return one staged algorithm definition by symbolic stage name."""
function buildup_algorithm(stage::S, layer::L) where {S<:Symbol,L<:IsingLearning.LayeredIsingGraphLayer}
    if stage === :free_phase_short
        return free_phase_short_algorithm(layer)
    elseif stage === :free_phase_full
        return input_field_free_phase_algorithm(layer)
    elseif stage === :free_then_nudged
        return free_then_nudged_algorithm(layer)
    elseif stage === :contrastive_static_beta
        return contrastive_static_beta_algorithm(layer)
    elseif stage === :contrastive_runtime_beta
        return input_field_contrastive_algorithm(layer)
    else
        error("Unsupported buildup stage `$stage`.")
    end
end

"""Create the common `_state` payload shared by all staged process cases."""
function buildup_state_init(graph::G, layer::L, input_hidden_w::R) where {
    G,
    L<:IsingLearning.LayeredIsingGraphLayer,
    R<:Base.RefValue,
}
    state = II.state(graph)
    return StatefulAlgorithms.Init(:_state;
        x = Ref(zeros(eltype(graph), INPUT_DIM)),
        y = Ref(zeros(eltype(graph), length(layer.output_layer))),
        input_hidden_w = input_hidden_w,
        buffers = input_field_gradient_buffer(graph, input_hidden_w[]),
        equilibrium_state = copy(state),
        nudged_state = similar(state),
    )
end

"""Create the common `dynamics` payload shared by all staged process cases."""
function buildup_dynamics_init(graph::G) where {G}
    return StatefulAlgorithms.Init(:dynamics, model = graph)
end

"""Build one normal `Process` for one staged diagnosis case."""
function buildup_process(algorithm::A, layer::L, source_graph::G, input_hidden_w::R) where {
    A,
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    R<:Base.RefValue,
}
    graph = shared_worker_graph(source_graph)
    return StatefulAlgorithms.Process(
        StatefulAlgorithms.resolve(algorithm),
        buildup_state_init(graph, layer, input_hidden_w),
        buildup_dynamics_init(graph);
        repeat = 1,
    )
end

"""Build one synchronous `InlineProcess` for one staged diagnosis case."""
function buildup_inline_process(algorithm::A, layer::L, source_graph::G, input_hidden_w::R) where {
    A,
    L<:IsingLearning.LayeredIsingGraphLayer,
    G,
    R<:Base.RefValue,
}
    graph = shared_worker_graph(source_graph)
    return StatefulAlgorithms.InlineProcess(
        StatefulAlgorithms.resolve(algorithm),
        buildup_state_init(graph, layer, input_hidden_w),
        buildup_dynamics_init(graph);
        repeats = 1,
    )
end

"""Load the first MNIST sample into one staged process case."""
function buildup_prime_sample!(proc::P, x::X, y::Y) where {P,X<:AbstractMatrix,Y<:AbstractMatrix}
    ctx = worker_context(proc)

    # Some staged cases only model the free phase, so only populate states that exist.
    hasproperty(ctx, :x) && copyto!(ctx.x[], view(x, :, 1))
    hasproperty(ctx, :y) && copyto!(ctx.y[], view(y, :, 1))
    return proc
end

"""Run one staged diagnosis case through the requested entrypoint."""
function buildup_run_case!(stage::S, entrypoint::E, setup, xtrain::X, ytrain::Y) where {
    S<:Symbol,
    E<:Symbol,
    X<:AbstractMatrix,
    Y<:AbstractMatrix,
}
    algorithm = buildup_algorithm(stage, setup.layer)
    input_hidden_w = Ref(copy(setup.input_hidden_w))

    if entrypoint === :process_run
        proc = buildup_process(algorithm, setup.layer, setup.graph, input_hidden_w)
        try
            buildup_prime_sample!(proc, xtrain, ytrain)
            StatefulAlgorithms.reset!(proc)
            run(proc)
            wait(proc)
            return nothing
        finally
            close(proc)
        end
    elseif entrypoint === :process_inline
        proc = buildup_process(algorithm, setup.layer, setup.graph, input_hidden_w)
        try
            buildup_prime_sample!(proc, xtrain, ytrain)
            StatefulAlgorithms.reset!(proc)
            StatefulAlgorithms.runprocessinline!(proc)
            return nothing
        finally
            close(proc)
        end
    elseif entrypoint === :inline_process
        proc = buildup_inline_process(algorithm, setup.layer, setup.graph, input_hidden_w)
        buildup_prime_sample!(proc, xtrain, ytrain)
        StatefulAlgorithms.reset!(proc)
        Base.run(proc)
        return nothing
    else
        error("Unsupported buildup entrypoint `$entrypoint`.")
    end
end

"""Write one child-case result file for the parent driver to read back."""
function write_buildup_case_result!(stage::S, entrypoint::E, status::AbstractString, seconds::Real, note::AbstractString) where {S,E}
    path = buildup_case_result_path(stage, entrypoint)
    open(path, "w") do io
        println(io, "stage=$(stage)")
        println(io, "entrypoint=$(entrypoint)")
        println(io, "status=$(status)")
        println(io, "seconds=$(Float64(seconds))")
        println(io, "note=$(replace(note, '\n' => ' '))")
    end
    return path
end

"""Read one child-case result file emitted by a staged subprocess."""
function read_buildup_case_result(path::P) where {P<:AbstractString}
    values = Dict{String,String}()
    for line in eachline(path)
        key, value = split(line, "="; limit = 2)
        values[key] = value
    end
    return (;
        stage = values["stage"],
        entrypoint = values["entrypoint"],
        status = values["status"],
        seconds = parse(Float64, values["seconds"]),
        note = values["note"],
    )
end

"""Append one staged-driver row to the aggregate results CSV."""
function append_buildup_row!(row::R) where {R<:NamedTuple}
    path = buildup_results_path()
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""Return one short human-readable note for a child process exit without a result file."""
function missing_buildup_note(exit_code::I) where {I<:Integer}
    if exit_code == 0
        return "child exited without writing a case result"
    else
        return "child exited with code $(exit_code) before writing a case result"
    end
end

"""Run one child Julia subprocess for one staged case and return its summary row."""
function run_buildup_child_case(stage::S, entrypoint::E) where {S<:Symbol,E<:Symbol}
    result_path = buildup_case_result_path(stage, entrypoint)
    stdout_path = buildup_stdout_path(stage, entrypoint)
    stderr_path = buildup_stderr_path(stage, entrypoint)
    rm(result_path; force = true)
    rm(stdout_path; force = true)
    rm(stderr_path; force = true)

    script_path = abspath(@__FILE__)
    julia_cmd = Base.julia_cmd()
    cmd = `$julia_cmd --project=ext/IsingLearning $script_path case $(String(stage)) $(String(entrypoint))`

    start_time = time()
    open(stdout_path, "w") do stdout_io
        open(stderr_path, "w") do stderr_io
            proc = run(pipeline(cmd; stdout = stdout_io, stderr = stderr_io); wait = false)

            # Run each stage in a fresh Julia so a hang cannot contaminate later cases.
            while process_running(proc) && (time() - start_time) < PROCESS_HANG_BUILDUP_TIMEOUT_SECONDS
                sleep(0.1)
            end

            if process_running(proc)
                kill(proc)
                try
                    wait(proc)
                catch
                end
                return (;
                    timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                    stage = String(stage),
                    entrypoint = String(entrypoint),
                    status = "timeout",
                    seconds = PROCESS_HANG_BUILDUP_TIMEOUT_SECONDS,
                    note = "timed out after $(PROCESS_HANG_BUILDUP_TIMEOUT_SECONDS)s",
                )
            end

            wait(proc)
            if isfile(result_path)
                result = read_buildup_case_result(result_path)
                return (;
                    timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                    stage = result.stage,
                    entrypoint = result.entrypoint,
                    status = result.status,
                    seconds = result.seconds,
                    note = result.note,
                )
            end

            exit_code = something(proc.exitcode, -999)
            return (;
                timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                stage = String(stage),
                entrypoint = String(entrypoint),
                status = exit_code == 0 ? "missing_result" : "crash",
                seconds = time() - start_time,
                note = missing_buildup_note(exit_code),
            )
        end
    end
end

"""Write a short markdown summary of the staged hang diagnosis."""
function write_buildup_summary!(rows::R) where {R<:AbstractVector}
    path = joinpath(PROCESS_HANG_BUILDUP_DIR, "summary.md")
    open(path, "w") do io
        println(io, "# Process Hang Buildup Summary")
        println(io)
        println(io, "- Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "- Timeout per child case: `", PROCESS_HANG_BUILDUP_TIMEOUT_SECONDS, " s`")
        println(io)

        # Report the first non-ok stage for each entrypoint in stage order.
        for entrypoint in PROCESS_HANG_BUILDUP_ENTRYPOINTS
            first_bad = nothing
            for stage in PROCESS_HANG_BUILDUP_STAGE_ORDER
                match = findfirst(row -> row.stage == String(stage) && row.entrypoint == String(entrypoint), rows)
                isnothing(match) && continue
                row = rows[match]
                if row.status != "ok"
                    first_bad = row
                    break
                end
            end

            println(io, "## `", entrypoint, "`")
            if isnothing(first_bad)
                println(io, "- All staged cases completed.")
            else
                println(io, "- First failing stage: `", first_bad.stage, "`")
                println(io, "- Status: `", first_bad.status, "`")
                println(io, "- Seconds: `", round(first_bad.seconds; digits = 3), "`")
                println(io, "- Note: ", first_bad.note)
            end
            println(io)
        end
    end
    return path
end

"""Run the staged diagnosis driver across all stages and entrypoints."""
function run_buildup_driver!()
    mkpath(PROCESS_HANG_BUILDUP_DIR)
    rm(buildup_results_path(); force = true)
    rows = NamedTuple[]

    for stage in PROCESS_HANG_BUILDUP_STAGE_ORDER
        for entrypoint in PROCESS_HANG_BUILDUP_ENTRYPOINTS
            row = run_buildup_child_case(stage, entrypoint)
            push!(rows, row)
            append_buildup_row!(row)
        end
    end

    write_buildup_summary!(rows)
    println("csv=" * buildup_results_path())
    println("summary=" * joinpath(PROCESS_HANG_BUILDUP_DIR, "summary.md"))
    return rows
end

"""Run one child case and persist a machine-readable result for the parent driver."""
function run_buildup_child_case!()
    length(ARGS) == 3 || error("Usage: process_hang_buildup.jl case <stage> <entrypoint>")
    stage = Symbol(ARGS[2])
    entrypoint = Symbol(ARGS[3])
    config = buildup_config()
    setup = build_layer(config)
    xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)

    started = time()
    try
        buildup_run_case!(stage, entrypoint, setup, xtrain, ytrain)
        write_buildup_case_result!(stage, entrypoint, "ok", time() - started, "completed")
    catch err
        note = sprint(showerror, err, catch_backtrace())
        write_buildup_case_result!(stage, entrypoint, "error", time() - started, note)
        rethrow()
    end
    return nothing
end

"""Dispatch either the subprocess case runner or the aggregate staged driver."""
function main()
    if !isempty(ARGS) && ARGS[1] == "case"
        run_buildup_child_case!()
    else
        run_buildup_driver!()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
