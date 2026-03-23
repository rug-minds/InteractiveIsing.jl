using InteractiveIsing
using InteractiveIsing.Processes
using InteractiveUtils
using Profile
using Random

const ALLOC_WG = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

function make_example_process(; side_length = 100, temperature = 2.0f0, steps = 100_000)
    g = IsingGraph(side_length, side_length, Discrete(), ALLOC_WG)
    temp!(g, temperature)
    algo = g.default_algorithm
    p = InlineProcess(algo, Input(algo; state = g); lifetime = steps)
    return (; graph = g, process = p)
end

runtime_context(p::InlineProcess) = Processes.merge_into_globals(p.context, (; process = p))
loop_algo(p::InlineProcess) = p.taskdata.func
named_algo(p::InlineProcess) = only(Processes.getalgos(loop_algo(p)))
context_view(p::InlineProcess) = view(runtime_context(p), named_algo(p))
metro_algo(p::InlineProcess) = Processes.getalgo(named_algo(p))

function metropolis_parts(p::InlineProcess)
    scv = context_view(p)
    metro = metro_algo(p)
    (; rng, state, hamiltonian, proposer, proposal) = scv
    return (; scv, metro, rng, state, hamiltonian, proposer, proposal)
end

run_call(p::InlineProcess) = run(p)
merge_globals_call(p::InlineProcess) = runtime_context(p)
generated_loop_call(p::InlineProcess) = Processes.generated_processloop(p, loop_algo(p), runtime_context(p), Processes.lifetime(p))
identifiable_step_call(p::InlineProcess) = Processes.step!(named_algo(p), runtime_context(p))
metropolis_step_call(p::InlineProcess) = Processes.step!(metro_algo(p), context_view(p))

function proposer_rand_call(p::InlineProcess)
    (; rng, proposer) = metropolis_parts(p)
    return rand(rng, proposer)
end

function deltaE_call(p::InlineProcess)
    (; rng, state, hamiltonian, proposer) = metropolis_parts(p)
    proposal = rand(rng, proposer)
    return InteractiveIsing.calculate(InteractiveIsing.ΔH(), hamiltonian, state, proposal)
end

function inject_call(p::InlineProcess)
    (; rng, proposer, scv) = metropolis_parts(p)
    proposal = rand(rng, proposer)
    return inject(scv, (; proposal))
end

function update_call(p::InlineProcess)
    (; rng, proposer, scv, metro, hamiltonian) = metropolis_parts(p)
    proposal = rand(rng, proposer)
    return InteractiveIsing.update!(metro, hamiltonian, scv.state, proposal)
end

function warm_breakdown!(p::InlineProcess)
    reset!(p); run_call(p)
    reset!(p); merge_globals_call(p)
    reset!(p); generated_loop_call(p)
    reset!(p); identifiable_step_call(p)
    reset!(p); metropolis_step_call(p)
    reset!(p); proposer_rand_call(p)
    reset!(p); deltaE_call(p)
    reset!(p); inject_call(p)
    reset!(p); update_call(p)
    return p
end

function print_inlineprocess_alloc_breakdown(p::InlineProcess)
    warm_breakdown!(p)

    reset!(p); println("run(p): ", @allocated run_call(p))
    reset!(p); println("merge_into_globals: ", @allocated merge_globals_call(p))
    reset!(p); println("generated_processloop: ", @allocated generated_loop_call(p))
    reset!(p); println("step!(named_algo, runtime_context): ", @allocated identifiable_step_call(p))
    reset!(p); println("step!(metropolis, context_view): ", @allocated metropolis_step_call(p))
    reset!(p); println("rand(rng, proposer): ", @allocated proposer_rand_call(p))
    reset!(p); println("calculate(ΔH(), hamiltonian, state, proposal): ", @allocated deltaE_call(p))
    reset!(p); println("inject(context_view, (; proposal)): ", @allocated inject_call(p))
    reset!(p); println("update!(metropolis, hamiltonian, state, proposal): ", @allocated update_call(p))
    return nothing
end

function alloc_profile_run!(p::InlineProcess; sample_rate = 1.0, format = :flat, print_kwargs...)
    reset!(p)
    run(p)
    Profile.Allocs.clear()
    reset!(p)
    Profile.Allocs.@profile sample_rate=sample_rate run(p)
    results = Profile.Allocs.fetch()
    Profile.Allocs.print(stdout, results; format, print_kwargs...)
    return results
end

function print_inlineprocess_llvm(p::InlineProcess)
    reset!(p)
    scv = context_view(p)
    metro = metro_algo(p)
    println("LLVM: run(::InlineProcess)")
    InteractiveUtils.code_llvm(stdout, run, Tuple{typeof(p)})
    println()
    println("LLVM: Processes.step!(::$(typeof(metro)), ::$(typeof(scv)))")
    InteractiveUtils.code_llvm(stdout, Processes.step!, Tuple{typeof(metro), typeof(scv)})
    return nothing
end
