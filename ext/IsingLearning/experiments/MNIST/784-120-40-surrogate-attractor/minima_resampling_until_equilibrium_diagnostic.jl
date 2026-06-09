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

"""Parse a comma-separated `Float32` list from the environment."""
function env_ft_list(name::S, default::AbstractVector{FT}) where {S<:AbstractString}
    raw = get(ENV, String(name), join(default, ","))
    return FT[parse(FT, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]
end

"""Parse a comma-separated string list from the environment."""
function env_string_list(name::S, default::AbstractVector{String}) where {S<:AbstractString}
    raw = get(ENV, String(name), join(default, ","))
    return String[strip(part) for part in split(raw, ",") if !isempty(strip(part))]
end

Base.@kwdef struct MinimaUntilEquilibriumConfig{T<:AbstractFloat,S<:AbstractString,VT<:AbstractVector{T},VS<:AbstractVector{S}}
    samples_per_class::Int = env_int("ISING_MNIST_MINUNTIL_SAMPLES_PER_CLASS", 1)
    base_max_sweeps::T = env_ft("ISING_MNIST_MINUNTIL_BASE_MAX_SWEEPS", 200)
    base_min_sweeps::T = env_ft("ISING_MNIST_MINUNTIL_BASE_MIN_SWEEPS", 25)
    post_max_sweeps::VT = env_ft_list("ISING_MNIST_MINUNTIL_POST_MAX_SWEEPS", FT[25, 50, 100])
    post_min_sweeps::T = env_ft("ISING_MNIST_MINUNTIL_POST_MIN_SWEEPS", 5)
    check_interval_sweeps::T = env_ft("ISING_MNIST_MINUNTIL_CHECK_INTERVAL_SWEEPS", 1)
    window_size::Int = env_int("ISING_MNIST_MINUNTIL_WINDOW_SIZE", 8)
    scale_samples::Int = env_int("ISING_MNIST_MINUNTIL_SCALE_SAMPLES", 8)
    relative_tolerance::T = env_ft("ISING_MNIST_MINUNTIL_REL_TOL", 0.05)
    absolute_tolerance::T = env_ft("ISING_MNIST_MINUNTIL_ABS_TOL", 1.0f-4)
    noise_factors::VT = env_ft_list("ISING_MNIST_MINUNTIL_NOISE_FACTORS", FT[0.5, 1, 2, 4])
    draws_per_variant::Int = env_int("ISING_MNIST_MINUNTIL_DRAWS", 20)
    modes::VS = env_string_list("ISING_MNIST_MINUNTIL_MODES", ["base", "chain"])
    checkpoint::S = get(ENV, "ISING_MNIST_MINUNTIL_CHECKPOINT", "")
    outdir::S = get(
        ENV,
        "ISING_MNIST_MINUNTIL_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", default_run_dirname("minima_resampling_until_equilibrium")),
    )
end

"""Return an empty energy-equilibrium statistics record."""
function minuntil_initial_stats(::Type{T}) where {T<:AbstractFloat}
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

"""Return whether an energy-equilibrium statistics record has met its criterion."""
function minuntil_done(stats::S) where {S}
    return Bool(stats.equilibrated)
end

"""Update a sliding full-energy-delta equilibrium heuristic after one chunk."""
function minuntil_energy_check!(
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

# Process adapter for the chunk-level full-energy equilibrium heuristic.
StatefulAlgorithms.@ProcessAlgorithm function MinUntilEnergyStats!(
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
    updated = minuntil_energy_check!(
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

"""Build one relaxation chunk plus equilibrium-check routine."""
function minuntil_chunk_check_algorithm(
    dynamics_algorithm::D,
    chunk_steps::I,
    min_checks::J,
    window_size::K,
    scale_samples::M,
    relative_tolerance::Float32,
    absolute_tolerance::Float32,
) where {D,I<:Integer,J<:Integer,K<:Integer,M<:Integer}
    return StatefulAlgorithms.@Routine begin
        @state eqstats = minuntil_initial_stats(FT)
        @state delta_window = zeros(FT, window_size)
        @state min_checks = min_checks
        @state window_size = window_size
        @state scale_samples = scale_samples
        @state relative_tolerance = relative_tolerance
        @state absolute_tolerance = absolute_tolerance
        @alias dynamics = dynamics_algorithm
        model = @repeat chunk_steps dynamics()
        eqstats = MinUntilEnergyStats!(
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

"""Build an until-equilibrium worker spec with a hard sweep cap."""
function minuntil_worker_spec(
    dynamics_algorithm::D,
    n_units::I,
    min_sweeps::T,
    max_sweeps::T,
    config::C,
) where {D,I<:Integer,T<:Real,C<:MinimaUntilEquilibriumConfig}
    chunk_steps = max(1, round(Int, config.check_interval_sweeps * n_units))
    min_checks = max(1, ceil(Int, min_sweeps / config.check_interval_sweeps))
    max_checks = max(1, ceil(Int, max_sweeps / config.check_interval_sweeps))
    algorithm = minuntil_chunk_check_algorithm(
        dynamics_algorithm,
        chunk_steps,
        min_checks,
        max(1, config.window_size),
        max(1, config.scale_samples),
        FT(config.relative_tolerance),
        FT(config.absolute_tolerance),
    )
    lifetime = StatefulAlgorithms.RepeatOrUntil(
        minuntil_done,
        max_checks,
        StatefulAlgorithms.Var(:_state, :eqstats),
    )
    return (; algorithm, lifetime, chunk_steps)
end

"""Construct one fresh inline worker for an until-equilibrium relaxation."""
function minuntil_worker(spec, graph::G) where {G}
    return StatefulAlgorithms.InlineProcess(
        StatefulAlgorithms.resolve(deepcopy(spec.algorithm)),
        StatefulAlgorithms.Init(:dynamics, model = graph);
        lifetime = spec.lifetime,
    )
end

"""Return the MNIST digit encoded in a repeated `-1/+1` target."""
function minuntil_target_digit(y::Y, output_replicas::I) where {Y<:AbstractVector,I<:Integer}
    replicas = Int(output_replicas)
    best_digit = 1
    best_score = typemin(FT)
    @inbounds for digit in 1:NCLASSES
        score = zero(FT)
        first_idx = (digit - 1) * replicas + 1
        for replica_idx in first_idx:(first_idx + replicas - 1)
            score += FT(y[replica_idx])
        end
        if score > best_score
            best_score = score
            best_digit = digit
        end
    end
    return best_digit - 1
end

"""Return the predicted digit from output replica scores."""
function minuntil_predicted_digit(state::S, output_idxs::O, output_replicas::I) where {
    S<:AbstractVector,
    O<:AbstractVector{Int},
    I<:Integer,
}
    replicas = Int(output_replicas)
    best_digit = 1
    best_score = typemin(FT)
    @inbounds for digit in 1:NCLASSES
        score = zero(FT)
        first_idx = (digit - 1) * replicas + 1
        for replica_idx in first_idx:(first_idx + replicas - 1)
            score += FT(state[output_idxs[replica_idx]])
        end
        score /= FT(replicas)
        if score > best_score
            best_score = score
            best_digit = digit
        end
    end
    return best_digit - 1
end

"""Return a stable sign-state hash used as a cheap minimum signature."""
function minuntil_signature(state::S) where {S<:AbstractVector}
    hash_value = UInt64(0xcbf29ce484222325)
    @inbounds for value in state
        hash_value ⊻= value >= zero(value) ? UInt64(0x01) : UInt64(0x00)
        hash_value *= UInt64(0x100000001b3)
    end
    return string(hash_value; base = 16, pad = 16)
end

"""Return the sign disagreement fraction between two states."""
function minuntil_distance(a::A, b::B) where {A<:AbstractVector,B<:AbstractVector}
    length(a) == length(b) || throw(ArgumentError("state lengths differ"))
    different = 0
    @inbounds for idx in eachindex(a, b)
        different += (a[idx] >= zero(a[idx])) != (b[idx] >= zero(b[idx]))
    end
    return different / length(a)
end

"""Reset one graph and install the projected MNIST input field."""
function minuntil_prepare_sample!(
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

"""Create an empty accumulator record for one combined diagnostic variant."""
function minuntil_accumulator()
    return (;
        count = Ref(0),
        correct = Ref(0),
        equilibrated = Ref(0),
        reward_sum = Ref(0.0),
        seconds_sum = Ref(0.0),
        sweeps_sum = Ref(0.0),
        energy_delta_sum = Ref(0.0),
        distance_base_sum = Ref(0.0),
        distance_previous_sum = Ref(0.0),
        signatures = Set{String}(),
    )
end

"""Update one per-variant accumulator from a sampled-minimum row."""
function minuntil_update_accumulator!(acc, row)
    acc.count[] += 1
    acc.correct[] += row.correct ? 1 : 0
    acc.equilibrated[] += row.equilibrated ? 1 : 0
    acc.reward_sum[] += Float64(row.reward)
    acc.seconds_sum[] += Float64(row.seconds)
    acc.sweeps_sum[] += Float64(row.sweeps)
    acc.energy_delta_sum[] += Float64(row.energy_delta)
    acc.distance_base_sum[] += Float64(row.distance_from_base)
    acc.distance_previous_sum[] += Float64(row.distance_from_previous)
    push!(acc.signatures, String(row.signature))
    return acc
end

"""Return one CSV-ready summary row for a combined diagnostic variant."""
function minuntil_summary_row(key, acc)
    count = max(acc.count[], 1)
    return (;
        mode = key.mode,
        post_max_sweeps = key.post_max_sweeps,
        noise_factor = key.noise_factor,
        noise_scale = key.noise_scale,
        draws = acc.count[],
        accuracy = acc.correct[] / count,
        equilibrated_fraction = acc.equilibrated[] / count,
        mean_reward = acc.reward_sum[] / count,
        mean_seconds = acc.seconds_sum[] / count,
        mean_sweeps = acc.sweeps_sum[] / count,
        mean_energy_delta = acc.energy_delta_sum[] / count,
        mean_distance_from_base = acc.distance_base_sum[] / count,
        mean_distance_from_previous = acc.distance_previous_sum[] / count,
        unique_signatures = length(acc.signatures),
        unique_fraction = length(acc.signatures) / count,
    )
end

"""Write one markdown settings sidecar for the combined diagnostic."""
function minuntil_write_settings!(path::P, train_config::C, config::D, noise_base::T) where {
    P<:AbstractString,
    C<:InputFieldMNISTConfig,
    D<:MinimaUntilEquilibriumConfig,
    T<:Real,
}
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Minima Resampling Until Equilibrium Diagnostic")
        println(io)
        println(io, "- architecture: `784-$(train_config.hidden)-$(NCLASSES * train_config.output_replicas)`")
        println(io, "- temp: `$(train_config.temp)`")
        println(io, "- stepsize: `$(train_config.stepsize)`")
        println(io, "- base min/max sweeps: `$(config.base_min_sweeps)` / `$(config.base_max_sweeps)`")
        println(io, "- post min sweeps: `$(config.post_min_sweeps)`")
        println(io, "- post max sweeps: `$(join(config.post_max_sweeps, ", "))`")
        println(io, "- check interval sweeps: `$(config.check_interval_sweeps)`")
        println(io, "- window size: `$(config.window_size)`")
        println(io, "- scale samples: `$(config.scale_samples)`")
        println(io, "- relative tolerance: `$(config.relative_tolerance)`")
        println(io, "- absolute tolerance: `$(config.absolute_tolerance)`")
        println(io, "- noise base: `$(noise_base)`")
        println(io, "- noise factors: `$(join(config.noise_factors, ", "))`")
        println(io, "- draws per variant: `$(config.draws_per_variant)`")
        println(io, "- modes: `$(join(config.modes, ", "))`")
        println(io, "- checkpoint: `$(isempty(config.checkpoint) ? "none" : config.checkpoint)`")
    end
    return path
end

"""Run minima resampling where both base and post-noise phases stop on equilibrium."""
function run_minima_resampling_until_equilibrium_diagnostic()
    config = MinimaUntilEquilibriumConfig{FT,String,Vector{FT},Vector{String}}()
    train_config = InputFieldMNISTConfig(;
        train_per_class = config.samples_per_class,
        test_per_class = config.samples_per_class,
        train_eval_per_class = config.samples_per_class,
        outdir = config.outdir,
    )
    setup = build_layer(train_config)
    input_hidden_w = Ref(setup.input_hidden_w)
    output_idxs = collect(Int, setup.layer.output_layer)
    output_replicas = length(output_idxs) ÷ NCLASSES
    n_units = setup.layer.nunits
    input_pattern = zeros(FT, n_units)
    noise_rng = Random.MersenneTwister(train_config.seed + 9917)
    noise_base = sqrt(max(FT(train_config.temp), zero(FT)))

    if !isempty(config.checkpoint)
        checkpoint = load_checkpoint(config.checkpoint)
        IsingLearning.sync_graph_params!(setup.graph, (; w = checkpoint.params.w, b = checkpoint.params.b))
        input_hidden_w[] = checkpoint.params.w_input
        println("[minuntil] loaded checkpoint $(config.checkpoint)")
    end

    mkpath(config.outdir)
    settings_path = minuntil_write_settings!(joinpath(config.outdir, "settings.md"), train_config, config, noise_base)
    rows_path = joinpath(config.outdir, "minima_resampling_until_samples.csv")
    summary_path = joinpath(config.outdir, "minima_resampling_until_summary.csv")
    rm(rows_path; force = true)
    rm(summary_path; force = true)

    println("[minuntil] loading balanced MNIST sample: $(config.samples_per_class)/class")
    xtrain, ytrain = balanced_mnist(:train, config.samples_per_class, train_config)
    base_spec = minuntil_worker_spec(
        setup.layer.validation_algorithm,
        n_units,
        config.base_min_sweeps,
        config.base_max_sweeps,
        config,
    )
    post_specs = Dict(
        FT(post_max_sweeps) => minuntil_worker_spec(
            setup.layer.validation_algorithm,
            n_units,
            config.post_min_sweeps,
            post_max_sweeps,
            config,
        )
        for post_max_sweeps in config.post_max_sweeps
    )
    accumulators = Dict{NamedTuple,Any}()

    println("[minuntil] settings: $(settings_path)")
    println("[minuntil] samples: $(rows_path)")
    println("[minuntil] summary: $(summary_path)")
    flush(stdout)

    for sample_idx in axes(xtrain, 2)
        x = view(xtrain, :, sample_idx)
        y = view(ytrain, :, sample_idx)
        label = minuntil_target_digit(y, output_replicas)

        minuntil_prepare_sample!(setup.graph, input_pattern, input_hidden_w[], x)
        base_worker = minuntil_worker(base_spec, setup.graph)
        base_seconds = @elapsed StatefulAlgorithms.run(base_worker; threaded = false)
        base_stats = StatefulAlgorithms.context(base_worker)._state.eqstats
        base_sweeps = Float64(base_stats.checks * base_spec.chunk_steps) / Float64(n_units)
        base_state = copy(II.state(setup.graph))
        base_energy = FT(base_stats.final_energy)
        base_signature = minuntil_signature(base_state)
        base_pred = minuntil_predicted_digit(base_state, output_idxs, output_replicas)
        base_reward = attractor_reward(y, base_state, output_idxs, output_replicas, train_config.reward_mode)

        append_row!(rows_path, (;
            sample = Int(sample_idx),
            label,
            mode = "base_relax_until",
            post_max_sweeps = FT(config.base_max_sweeps),
            noise_factor = FT(0),
            noise_scale = FT(0),
            draw = 0,
            seconds = round(base_seconds; digits = 6),
            checks = Int(base_stats.checks),
            sweeps = round(base_sweeps; digits = 3),
            equilibrated = Bool(base_stats.equilibrated),
            mean_abs_delta = FT(base_stats.mean_abs_delta),
            threshold = FT(base_stats.threshold),
            energy = base_energy,
            energy_delta = FT(0),
            pred = base_pred,
            correct = base_pred == label,
            reward = base_reward,
            distance_from_base = 0.0,
            distance_from_previous = 0.0,
            signature = base_signature,
        ))
        @printf(
            "[minuntil] sample=%d label=%d base_eq=%s base_sweeps=%.1f pred=%d energy=%.6g time=%.3fs\n",
            sample_idx,
            label,
            string(base_stats.equilibrated),
            base_sweeps,
            base_pred,
            base_energy,
            base_seconds,
        )
        flush(stdout)

        for mode in config.modes
            for post_max_sweeps in config.post_max_sweeps
                post_key = FT(post_max_sweeps)
                post_spec = post_specs[post_key]
                for noise_factor in config.noise_factors
                    noise_scale = FT(noise_base * noise_factor)
                    previous_state = copy(base_state)
                    key = (;
                        mode = String(mode),
                        post_max_sweeps = post_key,
                        noise_factor = FT(noise_factor),
                        noise_scale,
                    )
                    acc = get!(accumulators, key) do
                        minuntil_accumulator()
                    end

                    unique_local = Set{String}()
                    correct_local = 0
                    eq_local = 0
                    sweeps_local = 0.0
                    @inbounds for draw in 1:config.draws_per_variant
                        if mode == "base"
                            II.state(setup.graph) .= base_state
                            previous_state .= base_state
                        elseif mode != "chain"
                            throw(ArgumentError("unknown minima resampling mode: $(mode)"))
                        end

                        inject_gaussian_state_noise!(setup.graph, noise_rng, noise_scale)
                        post_worker = minuntil_worker(post_spec, setup.graph)
                        seconds = @elapsed StatefulAlgorithms.run(post_worker; threaded = false)
                        stats = StatefulAlgorithms.context(post_worker)._state.eqstats
                        sweeps = Float64(stats.checks * post_spec.chunk_steps) / Float64(n_units)
                        current_state = II.state(setup.graph)
                        energy = FT(stats.final_energy)
                        signature = minuntil_signature(current_state)
                        pred = minuntil_predicted_digit(current_state, output_idxs, output_replicas)
                        reward = attractor_reward(y, current_state, output_idxs, output_replicas, train_config.reward_mode)
                        distance_from_base = minuntil_distance(current_state, base_state)
                        distance_from_previous = minuntil_distance(current_state, previous_state)
                        row = (;
                            sample = Int(sample_idx),
                            label,
                            mode = String(mode),
                            post_max_sweeps = post_key,
                            noise_factor = FT(noise_factor),
                            noise_scale,
                            draw = Int(draw),
                            seconds = round(seconds; digits = 6),
                            checks = Int(stats.checks),
                            sweeps = round(sweeps; digits = 3),
                            equilibrated = Bool(stats.equilibrated),
                            mean_abs_delta = FT(stats.mean_abs_delta),
                            threshold = FT(stats.threshold),
                            energy,
                            energy_delta = energy - base_energy,
                            pred,
                            correct = pred == label,
                            reward,
                            distance_from_base,
                            distance_from_previous,
                            signature,
                        )
                        append_row!(rows_path, row)
                        minuntil_update_accumulator!(acc, row)
                        push!(unique_local, signature)
                        correct_local += row.correct ? 1 : 0
                        eq_local += row.equilibrated ? 1 : 0
                        sweeps_local += sweeps
                        previous_state .= current_state
                    end
                    @printf(
                        "[minuntil] sample=%d mode=%s cap=%.1f noise=%.2f acc=%.3f eq=%.3f unique=%d/%d mean_sweeps=%.2f\n",
                        sample_idx,
                        mode,
                        post_key,
                        noise_factor,
                        correct_local / config.draws_per_variant,
                        eq_local / config.draws_per_variant,
                        length(unique_local),
                        config.draws_per_variant,
                        sweeps_local / config.draws_per_variant,
                    )
                    flush(stdout)
                end
            end
        end
    end

    for key in sort(collect(keys(accumulators)); by = k -> (k.mode, k.post_max_sweeps, k.noise_factor))
        append_row!(summary_path, minuntil_summary_row(key, accumulators[key]))
    end

    println("[minuntil] completed")
    return (; rows_path, summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_minima_resampling_until_equilibrium_diagnostic()
end
