using InteractiveIsing
using InteractiveIsing.Processes
using InteractiveUtils

const LLVM_WG = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

function make_example_process(; side_length = 100, temperature = 2.0f0, steps = 100_000)
    g = IsingGraph(side_length, side_length, Discrete(), LLVM_WG)
    temp!(g, temperature)
    algo = g.default_algorithm
    p = InlineProcess(algo, Input(algo; state = g); lifetime = steps)
    return (; graph = g, process = p)
end

runtime_context(p::InlineProcess) = Processes.merge_into_globals(p.context, (; process = p))
loop_algo(p::InlineProcess) = p.taskdata.func

function save_llvm(func, args...; filename)
    open(filename, "w") do io
        code_llvm(io, func, typeof.(args); debuginfo = :none)
    end
    return filename
end

function save_typed(func, args...; filename)
    ci = code_typed(func, typeof.(args); optimize = true)
    open(filename, "w") do io
        for item in ci
            println(io, item)
            println(io)
        end
    end
    return filename
end

function dump_generated_processloop_llvm(; side_length = 100, temperature = 2.0f0, steps = 100_000, llvm_filename = "examples/generated_processloop_inline.ll", typed_filename = "examples/generated_processloop_inline_typed.txt")
    ex = make_example_process(; side_length, temperature, steps)
    p = ex.process
    run(p)
    reset!(p)
    algo = loop_algo(p)
    context = runtime_context(p)
    lt = Processes.lifetime(p)
    save_llvm(Processes.generated_processloop, p, algo, context, lt; filename = llvm_filename)
    save_typed(Processes.generated_processloop, p, algo, context, lt; filename = typed_filename)
    return (; process = p, llvm_filename, typed_filename)
end

