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

Base.@kwdef struct MinimaResamplingDiagnosticConfig{T<:AbstractFloat,S<:AbstractString,VT<:AbstractVector{T},VS<:AbstractVector{S}}
    samples_per_class::Int = env_int("ISING_MNIST_MINIMA_SAMPLES_PER_CLASS", 1)
    base_sweeps::T = env_ft("ISING_MNIST_MINIMA_BASE_SWEEPS", 200)
    post_sweeps::VT = env_ft_list("ISING_MNIST_MINIMA_POST_SWEEPS", FT[25, 50, 100])
    noise_factors::VT = env_ft_list("ISING_MNIST_MINIMA_NOISE_FACTORS", FT[0.5, 1, 2, 4])
    draws_per_variant::Int = env_int("ISING_MNIST_MINIMA_DRAWS", 20)
    modes::VS = env_string_list("ISING_MNIST_MINIMA_MODES", ["base", "chain"])
    checkpoint::S = get(ENV, "ISING_MNIST_MINIMA_CHECKPOINT", "")
    outdir::S = get(
        ENV,
        "ISING_MNIST_MINIMA_OUTDIR",
        joinpath(@__DIR__, "experiments", "current", default_run_dirname("minima_resampling_diagnostic")),
    )
end

"""Return the MNIST digit encoded in a repeated `-1/+1` output target."""
function target_digit(y::Y, output_replicas::I) where {Y<:AbstractVector,I<:Integer}
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

"""Return the predicted MNIST digit from output replica scores."""
function predicted_digit(state::S, output_idxs::O, output_replicas::I) where {
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
function sign_signature(state::S) where {S<:AbstractVector}
    hash_value = UInt64(0xcbf29ce484222325)
    @inbounds for value in state
        hash_value ⊻= value >= zero(value) ? UInt64(0x01) : UInt64(0x00)
        hash_value *= UInt64(0x100000001b3)
    end
    return string(hash_value; base = 16, pad = 16)
end

"""Return the sign disagreement fraction between two states."""
function sign_distance_fraction(a::A, b::B) where {A<:AbstractVector,B<:AbstractVector}
    length(a) == length(b) || throw(ArgumentError("state lengths differ"))
    different = 0
    @inbounds for idx in eachindex(a, b)
        different += (a[idx] >= zero(a[idx])) != (b[idx] >= zero(b[idx]))
    end
    return different / length(a)
end

"""Build a fixed-sweep relaxation routine for one already-prepared graph."""
function fixed_relax_algorithm(dynamics_algorithm::D, sweeps::T, n_units::I) where {
    D,
    T<:Real,
    I<:Integer,
}
    steps = max(1, round(Int, sweeps * n_units))
    return StatefulAlgorithms.@Routine begin
        @alias dynamics = dynamics_algorithm
        model = @repeat steps dynamics()
    end
end

"""Construct one reusable inline worker for a fixed-sweep relaxation routine."""
function fixed_relax_worker(algorithm::A, graph::G) where {A,G}
    return StatefulAlgorithms.InlineProcess(
        StatefulAlgorithms.resolve(deepcopy(algorithm)),
        StatefulAlgorithms.Init(:dynamics, model = graph);
        repeats = 1,
    )
end

"""Reset one graph and install the projected MNIST input field."""
function prepare_input_sample!(
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

"""Write a small markdown description of one minima-resampling diagnostic run."""
function write_minima_resampling_settings!(
    path::P,
    config::C,
    diagnostic::D,
    noise_base::T,
) where {P<:AbstractString,C<:InputFieldMNISTConfig,D<:MinimaResamplingDiagnosticConfig,T<:Real}
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Minima Resampling Diagnostic")
        println(io)
        println(io, "- architecture: `784-$(config.hidden)-$(NCLASSES * config.output_replicas)`")
        println(io, "- temp: `$(config.temp)`")
        println(io, "- stepsize: `$(config.stepsize)`")
        println(io, "- base relax sweeps: `$(diagnostic.base_sweeps)`")
        println(io, "- post-noise relax sweeps: `$(join(diagnostic.post_sweeps, ", "))`")
        println(io, "- noise base: `$(noise_base)`")
        println(io, "- noise factors: `$(join(diagnostic.noise_factors, ", "))`")
        println(io, "- draws per variant: `$(diagnostic.draws_per_variant)`")
        println(io, "- modes: `$(join(diagnostic.modes, ", "))`")
        println(io, "- checkpoint: `$(isempty(diagnostic.checkpoint) ? "none" : diagnostic.checkpoint)`")
    end
    return path
end

"""Create an empty accumulator record for one resampling variant."""
function variant_accumulator()
    return (;
        count = Ref(0),
        correct = Ref(0),
        reward_sum = Ref(0.0),
        seconds_sum = Ref(0.0),
        energy_delta_sum = Ref(0.0),
        distance_base_sum = Ref(0.0),
        distance_previous_sum = Ref(0.0),
        signatures = Set{String}(),
    )
end

"""Update one per-variant diagnostic accumulator from a sample row."""
function update_variant_accumulator!(acc, row)
    acc.count[] += 1
    acc.correct[] += row.correct ? 1 : 0
    acc.reward_sum[] += Float64(row.reward)
    acc.seconds_sum[] += Float64(row.seconds)
    acc.energy_delta_sum[] += Float64(row.energy_delta)
    acc.distance_base_sum[] += Float64(row.distance_from_base)
    acc.distance_previous_sum[] += Float64(row.distance_from_previous)
    push!(acc.signatures, String(row.signature))
    return acc
end

"""Return one CSV-ready summary row for a resampling variant."""
function variant_summary_row(key, acc)
    count = max(acc.count[], 1)
    return (;
        mode = key.mode,
        post_sweeps = key.post_sweeps,
        noise_factor = key.noise_factor,
        noise_scale = key.noise_scale,
        draws = acc.count[],
        accuracy = acc.correct[] / count,
        mean_reward = acc.reward_sum[] / count,
        mean_seconds = acc.seconds_sum[] / count,
        mean_energy_delta = acc.energy_delta_sum[] / count,
        mean_distance_from_base = acc.distance_base_sum[] / count,
        mean_distance_from_previous = acc.distance_previous_sum[] / count,
        unique_signatures = length(acc.signatures),
        unique_fraction = length(acc.signatures) / count,
    )
end

"""Run the perturb-and-short-relax minima resampling diagnostic."""
function run_minima_resampling_diagnostic()
    diagnostic = MinimaResamplingDiagnosticConfig{FT,String,Vector{FT},Vector{String}}()
    config = InputFieldMNISTConfig(;
        train_per_class = diagnostic.samples_per_class,
        test_per_class = diagnostic.samples_per_class,
        train_eval_per_class = diagnostic.samples_per_class,
        outdir = diagnostic.outdir,
    )
    setup = build_layer(config)
    input_hidden_w = Ref(setup.input_hidden_w)
    output_idxs = collect(Int, setup.layer.output_layer)
    output_replicas = length(output_idxs) ÷ NCLASSES
    n_units = setup.layer.nunits
    input_pattern = zeros(FT, n_units)
    noise_base = sqrt(max(FT(config.temp), zero(FT)))

    if !isempty(diagnostic.checkpoint)
        checkpoint = load_checkpoint(diagnostic.checkpoint)
        IsingLearning.sync_graph_params!(setup.graph, (; w = checkpoint.params.w, b = checkpoint.params.b))
        input_hidden_w[] = checkpoint.params.w_input
        println("[minima] loaded checkpoint $(diagnostic.checkpoint)")
    end

    mkpath(diagnostic.outdir)
    settings_path = write_minima_resampling_settings!(
        joinpath(diagnostic.outdir, "settings.md"),
        config,
        diagnostic,
        noise_base,
    )
    rows_path = joinpath(diagnostic.outdir, "minima_resampling_samples.csv")
    summary_path = joinpath(diagnostic.outdir, "minima_resampling_summary.csv")
    rm(rows_path; force = true)
    rm(summary_path; force = true)

    println("[minima] loading balanced MNIST sample: $(diagnostic.samples_per_class)/class")
    xtrain, ytrain = balanced_mnist(:train, diagnostic.samples_per_class, config)
    base_algorithm = fixed_relax_algorithm(setup.layer.validation_algorithm, diagnostic.base_sweeps, n_units)
    post_algorithms = Dict(
        FT(post_sweeps) => fixed_relax_algorithm(setup.layer.validation_algorithm, post_sweeps, n_units)
        for post_sweeps in diagnostic.post_sweeps
    )
    accumulators = Dict{NamedTuple,Any}()

    println("[minima] settings: $(settings_path)")
    println("[minima] samples: $(rows_path)")
    println("[minima] summary: $(summary_path)")
    flush(stdout)

    for sample_idx in axes(xtrain, 2)
        x = view(xtrain, :, sample_idx)
        y = view(ytrain, :, sample_idx)
        label = target_digit(y, output_replicas)

        prepare_input_sample!(setup.graph, input_pattern, input_hidden_w[], x)
        base_worker = fixed_relax_worker(base_algorithm, setup.graph)
        base_seconds = @elapsed StatefulAlgorithms.run(base_worker; threaded = false)
        base_state = copy(II.state(setup.graph))
        base_energy = FT(IsingLearning.graph_energy(setup.graph))
        base_signature = sign_signature(base_state)
        base_pred = predicted_digit(base_state, output_idxs, output_replicas)
        base_reward = attractor_reward(y, base_state, output_idxs, output_replicas, config.reward_mode)
        append_row!(rows_path, (;
            sample = Int(sample_idx),
            label,
            mode = "base_relax",
            post_sweeps = FT(0),
            noise_factor = FT(0),
            noise_scale = FT(0),
            draw = 0,
            seconds = round(base_seconds; digits = 6),
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
            "[minima] sample=%d label=%d base_pred=%d base_energy=%.6g base_time=%.3fs\n",
            sample_idx,
            label,
            base_pred,
            base_energy,
            base_seconds,
        )
        flush(stdout)

        for mode in diagnostic.modes
            for post_sweeps in diagnostic.post_sweeps
                post_key = FT(post_sweeps)
                post_worker = fixed_relax_worker(post_algorithms[post_key], setup.graph)
                for noise_factor in diagnostic.noise_factors
                    noise_scale = FT(noise_base * noise_factor)
                    previous_state = copy(base_state)
                    key = (;
                        mode = String(mode),
                        post_sweeps = post_key,
                        noise_factor = FT(noise_factor),
                        noise_scale,
                    )
                    acc = get!(accumulators, key) do
                        variant_accumulator()
                    end

                    unique_local = Set{String}()
                    correct_local = 0
                    seconds_local = 0.0
                    @inbounds for draw in 1:diagnostic.draws_per_variant
                        if mode == "base"
                            II.state(setup.graph) .= base_state
                            previous_state .= base_state
                        elseif mode != "chain"
                            throw(ArgumentError("unknown minima resampling mode: $(mode)"))
                        end

                        rng = StatefulAlgorithms.context(post_worker).dynamics.rng
                        inject_gaussian_state_noise!(setup.graph, rng, noise_scale)
                        seconds = @elapsed StatefulAlgorithms.run(post_worker; threaded = false)
                        current_state = II.state(setup.graph)
                        energy = FT(IsingLearning.graph_energy(setup.graph))
                        signature = sign_signature(current_state)
                        pred = predicted_digit(current_state, output_idxs, output_replicas)
                        reward = attractor_reward(
                            y,
                            current_state,
                            output_idxs,
                            output_replicas,
                            config.reward_mode,
                        )
                        distance_from_base = sign_distance_fraction(current_state, base_state)
                        distance_from_previous = sign_distance_fraction(current_state, previous_state)
                        row = (;
                            sample = Int(sample_idx),
                            label,
                            mode = String(mode),
                            post_sweeps = post_key,
                            noise_factor = FT(noise_factor),
                            noise_scale,
                            draw = Int(draw),
                            seconds = round(seconds; digits = 6),
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
                        update_variant_accumulator!(acc, row)
                        push!(unique_local, signature)
                        correct_local += row.correct ? 1 : 0
                        seconds_local += seconds
                        previous_state .= current_state
                    end
                    @printf(
                        "[minima] sample=%d mode=%s post=%.1f noise_factor=%.2f acc=%.3f unique=%d/%d mean_time=%.5fs\n",
                        sample_idx,
                        mode,
                        post_key,
                        noise_factor,
                        correct_local / diagnostic.draws_per_variant,
                        length(unique_local),
                        diagnostic.draws_per_variant,
                        seconds_local / diagnostic.draws_per_variant,
                    )
                    flush(stdout)
                end
            end
        end
    end

    for key in sort(collect(keys(accumulators)); by = k -> (k.mode, k.post_sweeps, k.noise_factor))
        append_row!(summary_path, variant_summary_row(key, accumulators[key]))
    end

    println("[minima] completed")
    return (; rows_path, summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_minima_resampling_diagnostic()
end
