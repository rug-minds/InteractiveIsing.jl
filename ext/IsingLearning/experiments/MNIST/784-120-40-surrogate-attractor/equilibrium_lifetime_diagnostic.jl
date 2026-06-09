include(joinpath(@__DIR__, "mnist_784_120_40_surrogate_attractor_adam.jl"))

using Printf

"""Read an environment variable as `Int`, with a concrete fallback."""
function env_int(name::S, default::I) where {S<:AbstractString,I<:Integer}
    return parse(Int, get(ENV, String(name), string(Int(default))))
end

"""Read an environment variable as `Float32`, with a concrete fallback."""
function env_ft(name::S, default::T) where {S<:AbstractString,T<:Real}
    return parse(FT, get(ENV, String(name), string(FT(default))))
end

Base.@kwdef struct EquilibriumDiagnosticConfig{T<:AbstractFloat,S<:AbstractString}
    samples_per_class::Int = env_int("ISING_MNIST_EQ_SAMPLES_PER_CLASS", 1)
    max_sweeps::T = env_ft("ISING_MNIST_EQ_MAX_SWEEPS", 200)
    min_sweeps::T = env_ft("ISING_MNIST_EQ_MIN_SWEEPS", 25)
    check_interval_sweeps::T = env_ft("ISING_MNIST_EQ_CHECK_INTERVAL_SWEEPS", 1)
    window_size::Int = env_int("ISING_MNIST_EQ_WINDOW_SIZE", 8)
    scale_samples::Int = env_int("ISING_MNIST_EQ_SCALE_SAMPLES", 8)
    relative_tolerance::T = env_ft("ISING_MNIST_EQ_REL_TOL", 0.05)
    absolute_tolerance::T = env_ft("ISING_MNIST_EQ_ABS_TOL", 1.0f-4)
    checkpoint::S = get(ENV, "ISING_MNIST_EQ_CHECKPOINT", "")
    outdir::S = get(
        ENV,
        "ISING_MNIST_EQ_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", default_run_dirname("equilibrium_lifetime_diagnostic")),
    )
end

"""Return the one-based MNIST digit encoded in a repeated `-1/+1` target."""
function target_digit(y::Y, output_replicas::I) where {Y<:AbstractVector,I<:Integer}
    replicas = Int(output_replicas)
    best_digit = 1
    best_score = typemin(FT)
    @inbounds for digit in 1:NCLASSES
        score = zero(FT)
        offset = (digit - 1) * replicas
        for replica_idx in 1:replicas
            score += FT(y[offset + replica_idx])
        end
        if score > best_score
            best_score = score
            best_digit = digit
        end
    end
    return best_digit - 1
end

"""Return an empty energy-equilibrium statistics record."""
function initial_equilibrium_stats(::Type{T}) where {T<:AbstractFloat}
    return (;
        previous_energy = T(NaN),
        window_index = 1,
        window_count = 0,
        scale_sum = zero(T),
        scale_count = 0,
        checks = 0,
        final_energy = T(NaN),
        last_abs_delta = T(Inf),
        mean_abs_delta = T(Inf),
        typical_delta = T(Inf),
        threshold = T(Inf),
        equilibrated = false,
    )
end

"""Return whether an equilibrium statistics record has met its stop criterion."""
function equilibrium_stats_done(stats::S) where {S}
    return Bool(stats.equilibrated)
end

"""Update one sliding energy-delta equilibrium test after a relaxation chunk."""
function energy_delta_equilibrium_check!(
    isinggraph::G,
    hamiltonian::H,
    previous_energy::T,
    delta_window::AbstractVector{T},
    window_index::Int,
    window_count::Int,
    scale_sum::T,
    scale_count::Int,
    checks::Int,
    min_checks::Int,
    window_size::Int,
    scale_samples::Int,
    relative_tolerance::T,
    absolute_tolerance::T,
) where {G,H,T<:AbstractFloat}
    energy = T(IsingLearning.graph_energy(isinggraph, hamiltonian))
    has_previous = isfinite(previous_energy)
    abs_delta = has_previous ? abs(energy - previous_energy) : T(Inf)

    if has_previous
        write_idx = window_index > window_size ? 1 : window_index
        delta_window[write_idx] = abs_delta
        window_index = write_idx == window_size ? 1 : write_idx + 1
        window_count = min(window_count + 1, window_size)
        if scale_count < scale_samples
            scale_sum += abs_delta
            scale_count += 1
        end
    end

    mean_abs_delta = zero(T)
    @inbounds for idx in 1:window_count
        mean_abs_delta += delta_window[idx]
    end
    window_count > 0 && (mean_abs_delta /= T(window_count))

    typical_delta = scale_count == 0 ? T(Inf) : max(scale_sum / T(scale_count), eps(T))
    threshold = max(absolute_tolerance, relative_tolerance * typical_delta)
    checks += 1
    enough_history = checks >= min_checks && window_count == window_size && scale_count == scale_samples
    equilibrated = enough_history && mean_abs_delta <= threshold

    return (;
        previous_energy = energy,
        window_index,
        window_count,
        scale_sum,
        scale_count,
        checks,
        final_energy = energy,
        last_abs_delta = abs_delta,
        mean_abs_delta,
        typical_delta,
        threshold,
        equilibrated,
    )
end

# Process adapter that returns one state variable for the lifetime selector.
StatefulAlgorithms.@ProcessAlgorithm function EnergyDeltaEquilibriumStats!(
    isinggraph::G,
    hamiltonian,
    eqstats,
    delta_window::AbstractVector{Float32},
    min_checks::Int,
    window_size::Int,
    scale_samples::Int,
    relative_tolerance::Float32,
    absolute_tolerance::Float32,
) where G
    updated = energy_delta_equilibrium_check!(
        isinggraph,
        hamiltonian,
        Float32(eqstats.previous_energy),
        delta_window,
        Int(eqstats.window_index),
        Int(eqstats.window_count),
        Float32(eqstats.scale_sum),
        Int(eqstats.scale_count),
        Int(eqstats.checks),
        min_checks,
        window_size,
        scale_samples,
        relative_tolerance,
        absolute_tolerance,
    )
    return (; eqstats = updated)
end

"""Build one reusable chunk-and-check routine for the lifetime diagnostic."""
function equilibrium_chunk_check_algorithm(
    dynamics_algorithm::D,
    chunk_steps::I,
    min_checks::J,
    window_size::K,
    scale_samples::M,
    relative_tolerance::Float32,
    absolute_tolerance::Float32,
) where {D,I<:Integer,J<:Integer,K<:Integer,M<:Integer}
    return StatefulAlgorithms.@Routine begin
        @state eqstats = initial_equilibrium_stats(FT)
        @state delta_window = zeros(FT, window_size)
        @state min_checks = min_checks
        @state window_size = window_size
        @state scale_samples = scale_samples
        @state relative_tolerance = relative_tolerance
        @state absolute_tolerance = absolute_tolerance
        @alias dynamics = dynamics_algorithm
        model = @repeat chunk_steps dynamics()
        eqstats = EnergyDeltaEquilibriumStats!(
            model,
            dynamics.hamiltonian,
            eqstats,
            delta_window,
            min_checks,
            window_size,
            scale_samples,
            relative_tolerance,
            absolute_tolerance,
        )
    end
end

"""Build a free relaxation diagnostic that stops by energy-delta equilibrium."""
function equilibrium_lifetime_algorithm(
    layer::L,
    config::C,
    diagnostic::D,
) where {L<:IsingLearning.LayeredIsingGraphLayer,C<:InputFieldMNISTConfig,D<:EquilibriumDiagnosticConfig}
    dynamics_algorithm = layer.validation_algorithm
    n_units = layer.nunits
    chunk_steps = max(1, round(Int, diagnostic.check_interval_sweeps * n_units))
    max_checks = max(1, ceil(Int, diagnostic.max_sweeps / diagnostic.check_interval_sweeps))
    min_checks = max(1, ceil(Int, diagnostic.min_sweeps / diagnostic.check_interval_sweeps))
    window_size = max(1, diagnostic.window_size)
    scale_samples = max(1, diagnostic.scale_samples)
    chunk_check_algorithm = equilibrium_chunk_check_algorithm(
        dynamics_algorithm,
        chunk_steps,
        min_checks,
        window_size,
        scale_samples,
        FT(diagnostic.relative_tolerance),
        FT(diagnostic.absolute_tolerance),
    )
    lifetime = StatefulAlgorithms.RepeatOrUntil(
        equilibrium_stats_done,
        max_checks,
        StatefulAlgorithms.Var(:_state, :eqstats),
    )
    return (; algorithm = chunk_check_algorithm, lifetime, check_interval_steps = chunk_steps)
end

"""Reset one graph and install one projected MNIST input field before a diagnostic run."""
function prepare_equilibrium_sample!(
    graph::G,
    input_pattern::P,
    input_hidden_w::W,
    x::X,
) where {G,P<:AbstractVector,W<:AbstractMatrix,X<:AbstractVector}
    II.resetstate!(graph)
    project_input_field_pattern!(input_pattern, input_hidden_w, x)
    install_input_field_pattern!(graph, input_pattern)
    return graph
end

"""Construct one diagnostic worker for a single MNIST example."""
function diagnostic_worker(
    algorithm::A,
    graph::G,
    lifetime::LT,
) where {A,G,LT}
    return StatefulAlgorithms.InlineProcess(
        StatefulAlgorithms.resolve(deepcopy(algorithm)),
        StatefulAlgorithms.Init(:dynamics, model = graph);
        lifetime,
    )
end

"""Write the diagnostic configuration to a small markdown sidecar."""
function write_diagnostic_settings!(path::P, config::C, diagnostic::D, relaxation_steps::I) where {
    P<:AbstractString,
    C<:InputFieldMNISTConfig,
    D<:EquilibriumDiagnosticConfig,
    I<:Integer,
}
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Equilibrium Lifetime Diagnostic")
        println(io)
        println(io, "- architecture: `784-$(config.hidden)-$(NCLASSES * config.output_replicas)`")
        println(io, "- temp: `$(config.temp)`")
        println(io, "- stepsize: `$(config.stepsize)`")
        println(io, "- original fixed relaxation steps: `$(relaxation_steps)`")
        println(io, "- max sweeps: `$(diagnostic.max_sweeps)`")
        println(io, "- min sweeps: `$(diagnostic.min_sweeps)`")
        println(io, "- check interval sweeps: `$(diagnostic.check_interval_sweeps)`")
        println(io, "- window size: `$(diagnostic.window_size)`")
        println(io, "- scale samples: `$(diagnostic.scale_samples)`")
        println(io, "- relative tolerance: `$(diagnostic.relative_tolerance)`")
        println(io, "- absolute tolerance: `$(diagnostic.absolute_tolerance)`")
        println(io, "- checkpoint: `$(isempty(diagnostic.checkpoint) ? "none" : diagnostic.checkpoint)`")
    end
    return path
end

"""Run the equilibrium diagnostic and write per-sample rows to CSV."""
function run_equilibrium_diagnostic()
    diagnostic = EquilibriumDiagnosticConfig{FT,String}()
    config = InputFieldMNISTConfig(;
        train_per_class = diagnostic.samples_per_class,
        test_per_class = diagnostic.samples_per_class,
        train_eval_per_class = diagnostic.samples_per_class,
        outdir = diagnostic.outdir,
    )
    setup = build_layer(config)
    input_hidden_w = Ref(setup.input_hidden_w)

    if !isempty(diagnostic.checkpoint)
        checkpoint = load_checkpoint(diagnostic.checkpoint)
        IsingLearning.sync_graph_params!(setup.graph, (; w = checkpoint.params.w, b = checkpoint.params.b))
        input_hidden_w[] = checkpoint.params.w_input
        println("[diagnostic] loaded checkpoint $(diagnostic.checkpoint)")
    end

    mkpath(diagnostic.outdir)
    settings_path = write_diagnostic_settings!(
        joinpath(diagnostic.outdir, "settings.md"),
        config,
        diagnostic,
        setup.relaxation_steps,
    )
    csv_path = joinpath(diagnostic.outdir, "equilibrium_lifetime_diagnostic.csv")
    rm(csv_path; force = true)

    println("[diagnostic] loading balanced MNIST sample: $(diagnostic.samples_per_class)/class")
    xtrain, ytrain = balanced_mnist(:train, diagnostic.samples_per_class, config)
    diagnostic_process = equilibrium_lifetime_algorithm(setup.layer, config, diagnostic)
    output_replicas = length(setup.layer.output_layer) ÷ NCLASSES
    n_units = setup.layer.nunits
    input_pattern = zeros(FT, n_units)
    check_interval_steps = diagnostic_process.check_interval_steps
    println("[diagnostic] settings: $(settings_path)")
    println("[diagnostic] output: $(csv_path)")
    flush(stdout)

    for sample_idx in axes(xtrain, 2)
        x = view(xtrain, :, sample_idx)
        y = view(ytrain, :, sample_idx)
        prepare_equilibrium_sample!(setup.graph, input_pattern, input_hidden_w[], x)
        worker = diagnostic_worker(diagnostic_process.algorithm, setup.graph, diagnostic_process.lifetime)
        seconds = @elapsed StatefulAlgorithms.run(worker; threaded = false)
        ctx = StatefulAlgorithms.context(worker)._state
        stats = ctx.eqstats
        sweeps = Float64(stats.checks * check_interval_steps) / Float64(n_units)
        row = (;
            timestamp = now(),
            sample = Int(sample_idx),
            label = target_digit(y, output_replicas),
            seconds = round(seconds; digits = 6),
            equilibrated = Bool(stats.equilibrated),
            checks = Int(stats.checks),
            sweeps = round(sweeps; digits = 3),
            final_energy = FT(stats.final_energy),
            last_abs_delta = FT(stats.last_abs_delta),
            mean_abs_delta = FT(stats.mean_abs_delta),
            typical_delta = FT(stats.typical_delta),
            threshold = FT(stats.threshold),
        )
        append_row!(csv_path, row)
        @printf(
            "[diagnostic] sample=%d label=%d eq=%s sweeps=%.1f checks=%d mean_delta=%.6g threshold=%.6g time=%.3fs\n",
            row.sample,
            row.label,
            string(row.equilibrated),
            row.sweeps,
            row.checks,
            row.mean_abs_delta,
            row.threshold,
            row.seconds,
        )
        flush(stdout)
    end

    return csv_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_equilibrium_diagnostic()
end
