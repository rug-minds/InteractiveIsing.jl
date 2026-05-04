using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using CairoMakie
using Dates
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

const FT = Float64
const MAX_FULLSWEEPS = parse(Int, get(ENV, "ISING_XOR_EQ_MAX_FULLSWEEPS", "20000"))
const SAMPLE_EVERY = parse(Int, get(ENV, "ISING_XOR_EQ_SAMPLE_EVERY", "100"))
const REPEATS = parse(Int, get(ENV, "ISING_XOR_EQ_REPEATS", "4"))
const HIDDEN_UNITS = parse(Int, get(ENV, "ISING_XOR_EQ_HIDDEN", "16"))
const OUTPUT_UNITS = parse(Int, get(ENV, "ISING_XOR_EQ_OUTPUT", "4"))
const WEIGHT_SEED = parse(Int, get(ENV, "ISING_XOR_EQ_WEIGHT_SEED", "2"))
const BIAS_SEED = parse(Int, get(ENV, "ISING_XOR_EQ_BIAS_SEED", "11"))
const BIAS_SCALE = parse(FT, get(ENV, "ISING_XOR_EQ_BIAS_SCALE", "0.1"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_EQ_WEIGHT_SCALE", "0.03"))
const WEIGHT_NORM = parse(FT, get(ENV, "ISING_XOR_EQ_WEIGHT_NORM", "0.2"))
const STEPSIZE = parse(FT, get(ENV, "ISING_XOR_EQ_STEPSIZE", "0.001"))
const BLOCK_SIZE = parse(Int, get(ENV, "ISING_XOR_EQ_BLOCK_SIZE", "8"))
const ADJUSTED = parse(Bool, lowercase(get(ENV, "ISING_XOR_EQ_ADJUSTED", "true")))
const INPUT_BIAS = parse(Bool, lowercase(get(ENV, "ISING_XOR_EQ_INPUT_BIAS", "true")))
const BASE_SEED = parse(Int, get(ENV, "ISING_XOR_EQ_BASE_SEED", "40000"))

const OUTDIR = get(
    ENV,
    "ISING_XOR_EQ_DIR",
    joinpath(@__DIR__, "..", "runs", "xor_equilibration_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
)

input_units() = INPUT_BIAS ? 17 : 16

function pattern_vertical()
    p = Matrix{FT}(undef, 4, 4)
    for row in 1:4, col in 1:4
        p[row, col] = col <= 2 ? -one(FT) : one(FT)
    end
    return vec(p)
end

function pattern_horizontal()
    p = Matrix{FT}(undef, 4, 4)
    for row in 1:4, col in 1:4
        p[row, col] = row <= 2 ? -one(FT) : one(FT)
    end
    return vec(p)
end

function xor_input(a::Bool, b::Bool)
    va = a ? pattern_vertical() : -pattern_vertical()
    hb = b ? pattern_horizontal() : -pattern_horizontal()
    x = zeros(FT, input_units())
    x[1:16] .= FT(0.5) .* (va .+ hb)
    INPUT_BIAS && (x[17] = one(FT))
    return x
end

function output_pattern(label::Bool)
    iseven(OUTPUT_UNITS) ||
        throw(ArgumentError("orthogonal output code requires an even output size"))
    pattern = ones(FT, OUTPUT_UNITS)
    if label
        split = OUTPUT_UNITS ÷ 2
        @inbounds for idx in (split + 1):OUTPUT_UNITS
            pattern[idx] = -one(FT)
        end
    end
    return pattern
end

readout_vector() = (output_pattern(true) .- output_pattern(false)) ./ FT(OUTPUT_UNITS)
readout_score(output) = dot(readout_vector(), output)

function small_weight_generator()
    rng = Random.MersenneTwister(WEIGHT_SEED)
    return AllToAllWeightGenerator((; dr, c1, c2, dc) -> WEIGHT_SCALE * randn(rng, FT))
end

function initial_bias_generator()
    rng = Random.MersenneTwister(BIAS_SEED)
    return g -> BIAS_SCALE .* randn(rng, FT, statelen(g))
end

function xor_graph()
    layers = (
        Layer(input_units(), StateSet(-one(FT), one(FT)), Continuous(), Coords(0, 1, 0)),
        Layer(HIDDEN_UNITS, StateSet(-one(FT), one(FT)), Continuous(), Coords(0, 2, 0)),
        Layer(OUTPUT_UNITS, StateSet(-one(FT), one(FT)), Continuous(), Coords(0, 3, 0)),
    )
    wg = small_weight_generator()
    bias = BIAS_SCALE == zero(FT) ?
        (g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))) :
        initial_bias_generator()
    local_potential = g -> InteractiveIsing.filltype(Vector, zero(FT), statelen(g))
    output_idxs = (input_units() + HIDDEN_UNITS + 1):(input_units() + HIDDEN_UNITS + OUTPUT_UNITS)
    hamiltonian =
        Quadratic(c = FT(0.5), localpotential = local_potential) +
        Quartic(c = zero(FT), localpotential = local_potential) +
        InteractiveIsing.Bilinear() +
        InteractiveIsing.MagField(b = bias) +
        LinearReadoutClamping(output_idxs, readout_vector(); β = zero(FT), target = zero(FT))

    graph = IsingGraph(
        layers[1], deepcopy(wg), layers[2], deepcopy(wg), layers[3],
        hamiltonian;
        precision = FT,
        index_set = g -> ToggledIndexSet(g),
    )
    diag(adj(graph)) .= zero(FT)
    return normalize_weights!(graph)
end

function normalize_weights!(graph)
    WEIGHT_NORM > zero(FT) || return graph
    vals = SparseArrays.getnzval(adj(graph))
    isempty(vals) && return graph
    rms = sqrt(sum(abs2, vals) / FT(length(vals)))
    isfinite(rms) && rms > zero(FT) || return graph
    vals .*= WEIGHT_NORM / rms
    return graph
end

function projected_residual(dH, s; α = one(FT), lo = -one(FT), hi = one(FT))
    projected = clamp(s - α * dH, lo, hi)
    return abs(projected - s) / α
end

function free_energy(model)
    s = InteractiveIsing.graphstate(model)
    b = InteractiveIsing.getparam(model.hamiltonian, InteractiveIsing.MagField, :b)
    return -FT(0.5) * dot(s, adj(model) * s) - dot(b, s)
end

function metrics(context)
    model = context.model
    active = collect(Iterators.flatten((layerrange(model[2]), layerrange(model[3]))))
    s = InteractiveIsing.graphstate(model)
    hamiltonian = context.hamiltonian

    dHs = Vector{FT}(undef, length(active))
    residuals = Vector{FT}(undef, length(active))
    @inbounds for (i, idx) in enumerate(active)
        dHs[i] = InteractiveIsing.calculate(InteractiveIsing.d_iH(), hamiltonian, model, idx)
        residuals[i] = projected_residual(dHs[i], s[idx])
    end

    active_state = s[active]
    out = state(model[3])
    return (;
        meanabs = mean(abs.(active_state)),
        frac_09 = count(>=(FT(0.9)), abs.(active_state)) / length(active_state),
        frac_099 = count(>=(FT(0.99)), abs.(active_state)) / length(active_state),
        frac_mid = count(<(FT(0.5)), abs.(active_state)) / length(active_state),
        dH_rms = sqrt(mean(abs2, dHs)),
        dH_max = maximum(abs.(dHs)),
        residual_rms = sqrt(mean(abs2, residuals)),
        residual_max = maximum(residuals),
        energy = free_energy(model),
        readout = readout_score(out),
        acceptance_rate = get(context, :acceptance_rate, NaN),
    )
end

function init_context(graph, x, seed)
    dynamics = BlockLangevin(
        stepsize = STEPSIZE,
        adjusted = ADJUSTED,
        block_size = BLOCK_SIZE,
        group_steps = 1,
    )
    InteractiveIsing.temp!(graph, FT(1e-4))
    context = Processes.init(dynamics, (; model = graph))
    Random.seed!(context.rng, seed)
    resetstate!(context.model)
    InteractiveIsing.off!(context.model.index_set, 1)
    state(context.model[1]) .= x
    return dynamics, context
end

function run_curve(case_idx, case, repeat_idx)
    graph = xor_graph()
    x = xor_input(case...)
    dynamics, context = init_context(graph, x, BASE_SEED + 10_000 * case_idx + repeat_idx)
    n_active = HIDDEN_UNITS + OUTPUT_UNITS
    steps_per_fullsweep = ceil(Int, n_active / BLOCK_SIZE)
    sample_sweeps = collect(0:SAMPLE_EVERY:MAX_FULLSWEEPS)
    sample_steps = sample_sweeps .* steps_per_fullsweep

    rows = NamedTuple[]
    next_sample_idx = 1
    for step_idx in 0:last(sample_steps)
        if step_idx == sample_steps[next_sample_idx]
            push!(rows, merge((; case, repeat = repeat_idx, fullsweeps = sample_sweeps[next_sample_idx], step = step_idx), metrics(context)))
            next_sample_idx += 1
            next_sample_idx > length(sample_steps) && break
        end
        context = merge(context, Processes.step!(dynamics, context))
    end
    return rows
end

function aggregate(rows)
    sweeps = sort(unique(row.fullsweeps for row in rows))
    return map(sweeps) do sweep
        subset = filter(row -> row.fullsweeps == sweep, rows)
        meanfield(name) = mean(getfield(row, name) for row in subset)
        (; fullsweeps = sweep,
           meanabs = meanfield(:meanabs),
           frac_09 = meanfield(:frac_09),
           frac_099 = meanfield(:frac_099),
           frac_mid = meanfield(:frac_mid),
           dH_rms = meanfield(:dH_rms),
           residual_rms = meanfield(:residual_rms),
           energy = meanfield(:energy),
           readout_abs = mean(abs(getfield(row, :readout)) for row in subset),
           acceptance_rate = meanfield(:acceptance_rate))
    end
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "fullsweeps,meanabs,frac_09,frac_099,frac_mid,dH_rms,residual_rms,energy,readout_abs,acceptance_rate")
        for row in rows
            @printf(
                io,
                "%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                row.fullsweeps,
                row.meanabs,
                row.frac_09,
                row.frac_099,
                row.frac_mid,
                row.dH_rms,
                row.residual_rms,
                row.energy,
                row.readout_abs,
                row.acceptance_rate,
            )
        end
    end
    return path
end

function plot_curve(path, rows)
    x = [row.fullsweeps for row in rows]
    fig = Figure(size = (980, 900))

    ax1 = Axis(fig[1, 1], xlabel = "free relaxation full sweeps", ylabel = "state extremeness")
    lines!(ax1, x, [row.meanabs for row in rows], label = "mean |s|", linewidth = 2)
    lines!(ax1, x, [row.frac_09 for row in rows], label = "frac |s| >= 0.9", linewidth = 2)
    lines!(ax1, x, [row.frac_mid for row in rows], label = "frac |s| < 0.5", linewidth = 2)
    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[2, 1], xlabel = "free relaxation full sweeps", ylabel = "fixed-point residual")
    lines!(ax2, x, [row.dH_rms for row in rows], label = "RMS dH/ds", linewidth = 2)
    lines!(ax2, x, [row.residual_rms for row in rows], label = "RMS projected residual", linewidth = 2)
    lines!(ax2, x, [row.acceptance_rate for row in rows], label = "acceptance rate", linewidth = 2)
    axislegend(ax2, position = :rt)

    ax3 = Axis(fig[3, 1], xlabel = "free relaxation full sweeps", ylabel = "energy / readout")
    lines!(ax3, x, [row.energy for row in rows], label = "H", linewidth = 2)
    axislegend(ax3, position = :rt)
    ax4 = Axis(fig[3, 1], yaxisposition = :right, ylabel = "|readout|")
    hidespines!(ax4, :l, :t, :b)
    hidexdecorations!(ax4)
    lines!(ax4, x, [row.readout_abs for row in rows], color = :darkorange, linewidth = 2)

    Label(fig[0, 1], "XOR free-phase relaxation diagnostics", fontsize = 24, font = :bold)
    save(path, fig)
    return path
end

function main()
    mkpath(OUTDIR)
    cases = ((false, false), (false, true), (true, false), (true, true))
    all_rows = NamedTuple[]

    println("Running XOR equilibration curve")
    println("  fullsweeps=$MAX_FULLSWEEPS sample_every=$SAMPLE_EVERY repeats=$REPEATS stepsize=$STEPSIZE block=$BLOCK_SIZE adjusted=$ADJUSTED")
    for (case_idx, case) in enumerate(cases), repeat_idx in 1:REPEATS
        println("  case=$case repeat=$repeat_idx")
        append!(all_rows, run_curve(case_idx, case, repeat_idx))
    end

    rows = aggregate(all_rows)
    csv_path = write_csv(joinpath(OUTDIR, "xor_equilibration.csv"), rows)
    plot_path = plot_curve(joinpath(OUTDIR, "xor_equilibration.png"), rows)
    println("Saved XOR equilibration CSV: $csv_path")
    println("Saved XOR equilibration plot: $plot_path")
end

main()
