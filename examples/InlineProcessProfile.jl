using InteractiveIsing
using InteractiveIsing.Processes
using Profile

const PROFILE_WG = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

function make_example_inlineprocess(; side_length = 100, temperature = 2.0f0, steps = 100_000)
    g = IsingGraph(side_length, side_length, Discrete(), PROFILE_WG)
    temp!(g, temperature)
    algo = g.default_algorithm
    p = InlineProcess(algo, Input(algo; state = g); lifetime = steps)
    return (; graph = g, process = p)
end

function run_inlineprocess!(p::InlineProcess; reset_before = true)
    reset_before && reset!(p)
    return run(p)
end

function alloc_profile_inlineprocess_run!(
    p::InlineProcess;
    sample_rate = 1.0,
    warmup = true,
    reset_before = true,
    io = stdout,
    print_kwargs...,
)
    if warmup
        run_inlineprocess!(p; reset_before = reset_before)
    end

    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=sample_rate run_inlineprocess!(p; reset_before = reset_before)
    results = Profile.Allocs.fetch()
    Profile.Allocs.print(io, results; print_kwargs...)
    return results
end

function cpu_profile_inlineprocess_run!(
    p::InlineProcess;
    warmup = true,
    reset_before = true,
    io = stdout,
    print_kwargs...,
)
    if warmup
        run_inlineprocess!(p; reset_before = reset_before)
    end

    Profile.clear()
    Profile.@profile run_inlineprocess!(p; reset_before = reset_before)
    Profile.print(io; print_kwargs...)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    example = make_example_inlineprocess()
    alloc_profile_inlineprocess_run!(example.process; format = :flat)
end
