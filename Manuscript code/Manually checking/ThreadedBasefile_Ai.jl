using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using Dates

if !isdefined(@__MODULE__, :ManuscriptTools)
    include(joinpath(@__DIR__, "..", "common", "ManuscriptTools.jl"))
end
const MT = ManuscriptTools

jobfield(job, key::Symbol, default) =
    hasproperty(job, key) ? getproperty(job, key) : default

mutable struct BasefileManagerState{Names,G,R,Result,Path}
    names::Names
    graphs::G
    runs::R
    results::Result
    paths::Path
end

function BasefileManagerState(njobs::Integer)
    slots = () -> Any[nothing for _ in 1:Int(njobs)]
    return BasefileManagerState(slots(), slots(), slots(), slots(), slots())
end

function basefile_process_inputs(g, run::MT.PulseRun, extra_inits = ())
    base_inputs = isnothing(run.Graph_Logger) ?
        (Init(run.M_Integrator, initialvalue = sum(MT.graph_array(g))),) :
        (
            Init(run.Graph_Logger, filepath = run.capture_dir),
            Init(run.M_Integrator, initialvalue = sum(MT.graph_array(g))),
        )
    return (base_inputs..., extra_inits...)
end

function basefile_process_inputs(g, run::MT.AnnealRun, extra_inits = ())
    return (
        Init(run.M_Integrator, initialvalue = sum(MT.graph_array(g))),
        extra_inits...,
    )
end

function build_basefile_job(
    params::MT.ManuscriptParams;
    name = Symbol("basefile_", params.xL, "x", params.yL, "x", params.zL),
    route::Symbol = :pulse,
    repeats::Integer = 1,
    weight_generator = nothing,
    extra_inits = (),
)
    graph = isnothing(weight_generator) ?
        MT.build_graph(params) :
        MT.build_graph(params; wg = weight_generator)
    run = if route == :pulse
        MT.build_pulse_process(graph, params)
    elseif route == :anneal
        MT.build_anneal_process(graph, params)
    else
        throw(ArgumentError("Unknown route $(repr(route)); use :pulse or :anneal."))
    end
    return (;
        name,
        params,
        route,
        targetgraph = graph,
        targetalgo = run.algorithm,
        run,
        inits = basefile_process_inputs(graph, run, extra_inits),
        repeats = Int(repeats),
    )
end

function make_basefile_worker(idx, manager::ProcessManager, job)
    graph = job.targetgraph
    algorithm = deepcopy(job.targetalgo)
    inputs = jobfield(job, :inits, ())
    repeats = jobfield(job, :repeats, 1)
    job_index = Int(job.index)

    manager.state.names[job_index] = jobfield(job, :name, Symbol("basefile_job_", idx))
    manager.state.graphs[job_index] = graph
    manager.state.runs[job_index] = jobfield(job, :run, job.targetalgo)

    graph_inputs = InteractiveIsing._mc_model_inits(algorithm, graph)
    return Process(algorithm, graph_inputs..., inputs...; repeats)
end

basefile_worker_name(idx, manager::ProcessManager, job) =
    jobfield(job, :name, Symbol("basefile_job_", idx))

function read_basefile_result(worker::Process, job, manager::ProcessManager)
    context = StatefulAlgorithms.context(worker)
    run = jobfield(job, :run, nothing)
    return isnothing(run) ? context : MT.read_run_context(context, run)
end

function consume_basefile_job!(slot::WorkerSlot, job, manager::ProcessManager)
    manager.state.results[Int(job.index)] =
        read_basefile_result(slot.worker, job, manager)
    return nothing
end

function indexed_basefile_jobs(jobs::AbstractVector)
    return [
        hasproperty(job, :index) ? job : merge((; index = i), job)
        for (i, job) in pairs(jobs)
    ]
end

function build_basefile_manager(
    jobs::AbstractVector;
    nworkers::Integer = min(length(jobs), max(1, Threads.nthreads())),
)
    recipe = (;
        makeworker = make_basefile_worker,
        workername = basefile_worker_name,
        afterjob! = consume_basefile_job!,
    )
    return ProcessManager(
        recipe;
        nworkers = Int(nworkers),
        state = BasefileManagerState(length(jobs)),
        worker_lifecycle = OnDemandWorkers(destroy_after_finalize = false),
        worker_type = Process,
        sync_policy = NoSync(),
        poll_interval = 0.0,
        job_type = eltype(jobs),
        result_type = Any,
    )
end

function save_basefile_results!(manager::ProcessManager, jobs::AbstractVector)
    for job in jobs
        i = Int(job.index)
        params = jobfield(job, :params, nothing)
        isnothing(params) && continue
        result = manager.state.results[i]
        route = jobfield(job, :route, :pulse)
        base_name = string(jobfield(job, :name, Symbol("basefile_job_", i)))
        manager.state.paths[i] = if route == :anneal
            MT.save_run_outputs(
                manager.state.graphs[i],
                params;
                anneal = result,
                base_name,
            )
        else
            MT.save_run_outputs(
                manager.state.graphs[i],
                params;
                pulse = result,
                base_name,
            )
        end
    end
    return manager.state.paths
end

function run_threaded_basefile!(
    jobs::AbstractVector;
    nworkers::Integer = min(length(jobs), max(1, Threads.nthreads())),
    schedule = Dynamic(),
    save_outputs::Bool = true,
)
    indexed_jobs = indexed_basefile_jobs(jobs)
    manager = build_basefile_manager(indexed_jobs; nworkers)
    runthreaded!(manager, indexed_jobs, schedule)
    wait(manager)
    save_outputs && save_basefile_results!(manager, indexed_jobs)
    return manager
end

function threaded_basefile_params(; root_outdir = joinpath(
    @__DIR__, "..", "..", "runs",
    "threaded_basefile_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"),
))
    common = (;
        xL = 10,
        yL = 10,
        zL = 10,
        JIsing = 1.0,
        Scale = 1.0,
        Screening = 1.0,
        Temp = 0.15f0,
        Temp_aneal = 2.0f0,
        time_fctr = 1.0,
        Steps_1 = 1200,
        Amp1 = 10.0,
        nrepeats = 3,
        proposal_delta = 0.1,
        a1 = -0.3,
        b1 = -2.1,
        c1 = 1.5,
        d1 = 0.0,
        e1 = 0.0,
        linear_field_coeff = 1.0,
        defect_field_coeff = 0.0,
        landau_mode = :independent,
        apply_weak_landau_disorder = false,
        disorder_seed = 1234,
        log_diagnostics = true,
        capture = false,
        save_figures = true,
        save_xlsx = true,
    )

    algorithms = (
        (:metropolis, (;)),
        (:local_langevin, (; stepsize = 0.02f0, adjusted = true)),
    )
    scales = (0.5, 1.0)

    # screenings = (0.1, 0.5, 1.0)

    params = MT.ManuscriptParams[]
    for (algorithm_name, algorithm_kwargs) in algorithms, 
        Scale in scales
        # Screening in screenings
        label = "$(algorithm_name)_Scale=$(Scale)"
        # label="$(algorithm_name)_Scale=$(Scale)_Screening=$(Screening)"
        push!(params, MT.ManuscriptParams(;
            common...,
            algorithm_name,
            algorithm_kwargs,
            Scale,
            outdir = joinpath(root_outdir, label),
        ))
    end

    return params
end

function threaded_basefile_jobs(paramsets = threaded_basefile_params())
    return [
        build_basefile_job(
            params;
            name = Symbol(
                "basefile_",
                params.algorithm_name,
                "_Scale_",
                replace(string(params.Scale), "." => "_"),
            ),
        )
        for params in paramsets
    ]
end

if abspath(PROGRAM_FILE) == @__FILE__
    jobs = threaded_basefile_jobs()
    manager = run_threaded_basefile!(jobs; nworkers = min(length(jobs), Threads.nthreads()))
    println("Finished threaded Basefile jobs.")
    for (i, path_info) in pairs(manager.state.paths)
        println("job ", i, " output: ", path_info)
    end
end


# p1 = MT.ManuscriptParams(
#     xL = 10,
#     yL = 10,
#     zL = 10,
#     Steps_1 = 1200,
#     nrepeats = 3,
#     Scale = 0.5,
#     outdir = raw"D:\data\experiment1",
# )

# p2 = MT.update_params(
#     p1;
#     Scale = 1.0,
#     outdir = raw"D:\data\experiment2",
# )

# jobs = [
#     build_basefile_job(p1; name = :scale_05),
#     build_basefile_job(p2; name = :scale_10),
# ]