using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

function isingweights(; dr)
    return dr == 1 ? 1.0f0 : 0.0f0
end

function idle_process(g, algo; repeat = Inf)
    local_algo = deepcopy(algo)
    inputs = InteractiveIsing._mc_model_inits(local_algo, g)
    p = Process(local_algo, inputs...; repeat)
    push!(processes(g), p)
    return p
end

initial_T = 2.0f0

wg = @WG isingweights NN=1

template_graph = IsingGraph(
    128, 128,
    wg;
    precision = Float32,
    periodic = (:x, :y),
)
temp!(template_graph, initial_T)

old_graph = deepcopy(template_graph)
injector_graph = deepcopy(template_graph)

old_process = idle_process(old_graph, Metropolis())

injector_algorithm = CompositeAlgorithm(
    Injector(),
    :metro => metropolis_nointeract(T = initial_T),
)
injector_process = idle_process(injector_graph, injector_algorithm)

injector_T = view(getcontext(injector_process), Var(:metro, :T))

# Run one of these manually while benchmarking.
interface(old_graph)
run(old_process)
interface(injector_graph)
run(injector_process)

# Change the injected Metropolis temperature while injector_process is running.
# injector_T[] = 2.5f0
