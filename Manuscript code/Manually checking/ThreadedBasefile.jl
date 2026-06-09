using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using Dates

include(joinpath(@__DIR__, "..", "common", "ManuscriptTools.jl"))
const MT = ManuscriptTools

###############################################################################
# 中文说明
#
# 这个文件是 Basefile.jl 的多线程模板。核心思想是：
# 1. 每个 job 只放已经构造好的 targetgraph、targetalgo 和 inits。
# 2. ProcessManager 只负责调度；真正的 Process 在 makeworker 中按 job 新建。
# 3. OnDemandWorkers(destroy_after_finalize = false) 会让每个 job 拿到一个新的
#    worker，不复用上一个实验的 graph/context。实验结束后 consume! 读出结果，
#    worker 留在 manager.slots 里，方便检查。
#
# 运行方式示例：
#     include("Manuscript code/Manually checking/ThreadedBasefile.jl")
#     manager = run_threaded_basefile!(jobs; nworkers = length(jobs))
#
# nworkers 控制“同时运行多少个 job”，不是 job 总数。比如 jobs 有 20 个，
# nworkers = 4 时会同时跑 4 个；每完成一个，新的 job 会创建新的 worker。
###############################################################################

"""
    jobfield(job, key, default)

读取 job NamedTuple 的可选字段。jobs 推荐形状：
`(; name, targetgraph, targetalgo, inits)`.
"""
function jobfield(job::J, key::Symbol, default) where {J}
    return hasproperty(job, key) ? getproperty(job, key) : default
end

"""
    BasefileManagerState(njobs)

ProcessManager 的共享状态。每个 job 的 name/graph/run/result/path 都按 job.index
写入固定位置，避免多个线程 push! 同一个数组。
"""
mutable struct BasefileManagerState{Names,G,R,Result,Path}
    names::Names
    graphs::G
    runs::R
    results::Result
    paths::Path
end

function BasefileManagerState(njobs::I) where {I<:Integer}
    n = Int(njobs)
    return BasefileManagerState(
        Any[nothing for _ in 1:n],
        Any[nothing for _ in 1:n],
        Any[nothing for _ in 1:n],
        Any[nothing for _ in 1:n],
        Any[nothing for _ in 1:n],
    )
end

"""
    build_basefile_pulse_run(; dynamics, fullsweep, Steps_1, kwargs...)

创建 Basefile 风格的完整 pulse + relax routine。这个函数返回的对象就是
job 里的 `targetalgo`，也就是传给 `Process(...)` 的完整 LoopAlgorithm/Routine，
不是单独的 Metropolis/Langevin dynamics。graph 本身只放在 job.targetgraph；
这里不接收 graph，避免 job 里出现额外的临时变量。
"""
function build_basefile_pulse_run(
    ;
    dynamics::D,
    fullsweep::F,
    Steps_1::S,
    time_fctr::T = 1.0,
    nrepeats::N = 3,
    Amp1::A = 10.0,
    capture::C = false,
    capture_dir::Path = joinpath(base_outdir, "capture"),
) where {D,F<:Integer,S<:Integer,T<:Real,N<:Integer,A<:Real,C<:Bool,Path<:AbstractString}
    Steps_1 % (4 * nrepeats) == 0 || throw(ArgumentError(
        "`Steps_1` must be divisible by `4 * nrepeats` for TrianglePulseA; got Steps_1 = $(Steps_1), nrepeats = $(nrepeats).",
    ))

    point_repeat = max(1, round(Int, fullsweep * time_fctr))
    pulse_time = max(1, round(Int, fullsweep * time_fctr * Steps_1))
    relax_time = max(1, round(Int, fullsweep * time_fctr * Steps_1 / 2))
    capture_interval1 = max(1, round(Int, pulse_time / (nrepeats * 4)))
    capture_interval2 = max(1, round(Int, relax_time / 2))

    pulse = MT.TrianglePulseA(Amp1, nrepeats)
    M_Integrator = Integrator(Float32, name = :M_integrator)
    M_Logger = Logger(Float32, name = :M_logger)
    B_Logger = MT.ValueLogger(:b)
    Graph_Logger = MT.ImageCapture(:Graph, -1.5, 1.5)
    target_dynamics = deepcopy(dynamics)
    graph_logger = capture ? Graph_Logger : nothing

    # 主动力学：每一步跑一次算法，每 point_repeat 记录一次 P 和外场。
    metro_pulse = @CompositeAlgorithm begin
        @alias dynamics = target_dynamics
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

    if capture
        # pulse 阶段改变外场，并按 capture_interval1 保存状态图。
        pulse_part = @CompositeAlgorithm begin
            @alias pulse = pulse
            @alias graph_logger = graph_logger
            @context metro_pulse = metro_pulse()

            @every point_repeat pulse(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
            @every capture_interval1 graph_logger(
                array = @transform(MT.graph_array, metro_pulse.dynamics.model),
            )
        end

        # relax 阶段不再改变外场，只继续动力学和可选截图。
        relax_part = @CompositeAlgorithm begin
            @alias graph_logger = graph_logger
            @context metro_pulse = metro_pulse()

            @every capture_interval2 graph_logger(
                array = @transform(MT.graph_array, metro_pulse.dynamics.model),
            )
        end
    else
        pulse_part = @CompositeAlgorithm begin
            @alias pulse = pulse
            @context metro_pulse = metro_pulse()

            @every point_repeat pulse(
                hamiltonian = metro_pulse.dynamics.hamiltonian,
                M = metro_pulse.dynamics.M,
            )
        end

        relax_part = @CompositeAlgorithm begin
            @context metro_pulse = metro_pulse()
        end
    end

    algorithm = @Routine begin
        @repeat pulse_time pulse_part()
        @repeat relax_time relax_part()
    end

    return algorithm
end

"""
    basefile_process_inputs(g, run[, extra_inits]) -> Tuple

为这个文件里的示例 pulse/anneal run 生成 `Process(...)` 初始化输入。
如果用户用 `MT.build_pulse_process` / `MT.build_anneal_process` 得到 `PulseRun`
或 `AnnealRun`，可以用这个 helper 生成 `Init(...)`。本文件的 job 也可以
完全不用这个 helper，直接在 `inits` 里传自己的初始化输入。
"""
function basefile_process_inputs(g::G, run::R, extra_inits = ()) where {G,R<:MT.PulseRun}
    base_inputs = isnothing(run.Graph_Logger) ?
        (Init(run.M_Integrator, initialvalue = sum(MT.graph_array(g))),) :
        (
            Init(run.Graph_Logger, filepath = run.capture_dir),
            Init(run.M_Integrator, initialvalue = sum(MT.graph_array(g))),
        )

    return (base_inputs..., extra_inits...)
end

function basefile_process_inputs(g::G, run::R, extra_inits = ()) where {G,R<:MT.AnnealRun}
    base_inputs = (Init(run.M_Integrator, initialvalue = sum(MT.graph_array(g))),)
    return (base_inputs..., extra_inits...)
end

"""
    make_basefile_worker(idx, manager, job) -> Process

OnDemandWorkers 会为每个 job 调用这里。job 必须提供：
- `name`: worker/process 名称，方便运行后检查；
- `targetgraph`: 已经构造好的 graph；
- `targetalgo`: 已经构造好的完整 Routine/Composite/LoopAlgorithm；
- `inits`: 传给 `Process` 的 Init/Override tuple。
"""
function make_basefile_worker(idx::I, manager::M, job::J) where {I<:Integer,M<:ProcessManager,J}
    graph = job.targetgraph
    targetalgo = job.targetalgo
    inputs = jobfield(job, :inits, ())
    repeats = jobfield(job, :repeats, 1)
    worker_name = jobfield(job, :name, Symbol("basefile_job_", idx))

    # 保存 graph/algo，consume! 和运行后检查时会用到。
    job_index = Int(job.index)
    manager.state.names[job_index] = worker_name
    manager.state.graphs[job_index] = graph
    manager.state.runs[job_index] = targetalgo

    algorithm = deepcopy(targetalgo)
    graph_inputs = InteractiveIsing._mc_model_inits(algorithm, graph)
    return Process(algorithm, graph_inputs..., inputs...; repeats)
end

"""
    basefile_worker_name(idx, manager, job) -> Symbol

ProcessManager 的 `workername` 回调。OnDemandWorkers 创建 worker 后会把这个
名字写进 `slot.name`，方便之后检查 `manager.slots`。
"""
function basefile_worker_name(idx::I, manager::M, job::J) where {I<:Integer,M<:ProcessManager,J}
    return jobfield(job, :name, Symbol("basefile_job_", idx))
end

"""
    read_basefile_result(worker, run) -> NamedTuple

从已完成 worker 的 context 中读出实验曲线。读取发生在 worker 删除之前。
"""
function read_basefile_result(worker::W, job::J, manager::M) where {W<:Process,J,M<:ProcessManager}
    context = StatefulAlgorithms.context(worker)
    reader = jobfield(job, :readresult, nothing)
    isnothing(reader) && return context
    return reader(context, worker, job, manager)
end

"""
    consume_basefile_job!(slot, job, manager)

ProcessManager 的 consume! 回调。把 worker context 里的结果搬到
manager.state.results[job.index]，避免 worker 删除后找不到结果。
"""
function consume_basefile_job!(slot::S, job::J, manager::M) where {S<:WorkerSlot,J,M<:ProcessManager}
    job_index = Int(job.index)
    manager.state.results[job_index] = read_basefile_result(slot.worker, job, manager)
    return nothing
end

"""
    build_basefile_manager(jobs; nworkers=min(length(jobs), Threads.nthreads()))

创建使用 OnDemandWorkers 的 ProcessManager。这个 manager 不复用实验 worker；
每个 job 都会通过 `make_basefile_worker` 新建一个 Process。这里不销毁
完成后的 worker，方便运行结束后检查 `manager.slots`。
"""
function build_basefile_manager(
    jobs::J;
    nworkers::N = min(length(jobs), max(1, Threads.nthreads())),
) where {J<:AbstractVector,N<:Integer}
    state = BasefileManagerState(length(jobs))
    recipe = (;
        makeworker = make_basefile_worker,
        workername = basefile_worker_name,
        afterjob! = consume_basefile_job!,
    )

    return ProcessManager(
        recipe;
        nworkers = Int(nworkers),
        state,
        worker_lifecycle = OnDemandWorkers(destroy_after_finalize = false),
        worker_type = Process,
        sync_policy = NoSync(),
        poll_interval = 0.0,
        job_type = eltype(jobs),
        result_type = Any,
    )
end

"""
    save_basefile_results!(manager, jobs) -> manager.state.paths

所有线程完成以后，在主线程串行保存输出。Makie/XLSX 写文件不放在 worker
线程里做，避免多个线程同时写图或写 Excel。
"""
function save_basefile_results!(manager::M, jobs::J) where {M<:ProcessManager,J<:AbstractVector}
    error("Generic targetgraph/targetalgo jobs do not have a built-in saver. Add a save hook or inspect manager.state.results.")
end

"""
    run_threaded_basefile!(jobs; nworkers=..., schedule=Dynamic(), save_outputs=false)

运行 jobs 向量并返回保持打开的 `ProcessManager`。结果在 `manager.state`
里面，worker 留在 `manager.slots` 里面。检查完以后手动调用 `close(manager)`。
"""
function indexed_basefile_jobs(jobs::J) where {J<:AbstractVector}
    return [
        hasproperty(job, :index) ? job : merge((; index = i), job)
        for (i, job) in pairs(jobs)
    ]
end

function run_threaded_basefile!(
    jobs::J;
    nworkers::N = min(length(jobs), max(1, Threads.nthreads())),
    schedule::S = Dynamic(),
    save_outputs::B = false,
) where {J<:AbstractVector,N<:Integer,S,B<:Bool}
    indexed_jobs = indexed_basefile_jobs(jobs)
    manager = build_basefile_manager(indexed_jobs; nworkers)

    runthreaded!(manager, indexed_jobs, schedule)
    save_outputs && save_basefile_results!(manager, indexed_jobs)
    return manager
end

###############################################################################
# 用户配置区
###############################################################################

run_stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
base_outdir = joinpath(@__DIR__, "..", "..", "runs", "threaded_basefile_" * run_stamp)

"""
    threaded_basefile_jobs() -> Vector{NamedTuple}

小型 job 模板。复制/修改这里即可增加 sweep：
- 每个 job entry 直接创建 `name`、`targetgraph`、`targetalgo` 和 `inits`；
- `make_basefile_worker` 不知道 graph/algo 是怎么创建的；
- 可选 `repeats = ...` 设置 `Process(...; repeats = repeats)`。
"""
function threaded_basefile_jobs()
    # 这里定义可复用的 dynamics；它们不是 job.targetalgo。
    # job.targetalgo 是下面 build_basefile_pulse_run(...) 返回的完整 routine。
    # nrepeats = 3 时，Steps_1 要是 12 的倍数，否则 TrianglePulseA 的四段 pulse 分不整。
    metropolis_dynamics = Metropolis()
    langevin_dynamics = LocalLangevin(stepsize = 0.02f0, adjusted = true)

    return [
        (;
            name = :basefile_10x10x8_metropolis,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "10x10x8_metropolis"),
                xL = 10,
                yL = 10,
                zL = 8,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = metropolis_dynamics,
                fullsweep = 10 * 10 * 8,
                Steps_1 = 804,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_10x10x8_langevin,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "10x10x8_langevin"),
                xL = 10,
                yL = 10,
                zL = 8,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = langevin_dynamics,
                fullsweep = 10 * 10 * 8,
                Steps_1 = 504,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_12x12x8_metropolis,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "12x12x8_metropolis"),
                xL = 12,
                yL = 12,
                zL = 8,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = metropolis_dynamics,
                fullsweep = 12 * 12 * 8,
                Steps_1 = 1008,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_12x12x8_langevin,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "12x12x8_langevin"),
                xL = 12,
                yL = 12,
                zL = 8,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = langevin_dynamics,
                fullsweep = 12 * 12 * 8,
                Steps_1 = 696,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_14x14x10_metropolis,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "14x14x10_metropolis"),
                xL = 14,
                yL = 14,
                zL = 10,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = metropolis_dynamics,
                fullsweep = 14 * 14 * 10,
                Steps_1 = 1200,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_14x14x10_langevin,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "14x14x10_langevin"),
                xL = 14,
                yL = 14,
                zL = 10,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = langevin_dynamics,
                fullsweep = 14 * 14 * 10,
                Steps_1 = 900,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_15x15x10_metropolis,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "15x15x10_metropolis"),
                xL = 15,
                yL = 15,
                zL = 10,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = metropolis_dynamics,
                fullsweep = 15 * 15 * 10,
                Steps_1 = 1404,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),

        (;
            name = :basefile_15x15x10_langevin,
            targetgraph = MT.build_graph(MT.ManuscriptParams(;
                outdir = joinpath(base_outdir, "15x15x10_langevin"),
                xL = 15,
                yL = 15,
                zL = 10,
                JIsing = 1.0,
                Scale = 1.0,
                Screening = 1.0,
                Temp = 0.15f0,
                proposal_delta = 0.1,
                a1 = -0.3,
                b1 = -2.1,
                c1 = 1.5,
            )),
            targetalgo = build_basefile_pulse_run(;
                dynamics = langevin_dynamics,
                fullsweep = 15 * 15 * 10,
                Steps_1 = 1092,
                nrepeats = 3,
            ),
            inits = (),
            repeats = 1,
        ),
    ]
end

jobs = threaded_basefile_jobs()
manager = run_threaded_basefile!(jobs; nworkers = length(jobs))
state = manager.state
wait(manager)

println("Finished threaded Basefile jobs.")
for (i, path_info) in pairs(state.paths)
    println("job ", i, " output: ", path_info)
end
