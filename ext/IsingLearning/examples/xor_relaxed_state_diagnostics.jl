using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Random
using SparseArrays
using LinearAlgebra
using Statistics
using Printf

const FT = Float64
const REPEATS = parse(Int, get(ENV, "ISING_XOR_DIAG_REPEATS", "50"))
const RELAXATION_STEPS = parse(Int, get(ENV, "ISING_XOR_DIAG_RELAXATION", "100"))
const HIDDEN_UNITS = parse(Int, get(ENV, "ISING_XOR_DIAG_HIDDEN", "16"))
const OUTPUT_UNITS = parse(Int, get(ENV, "ISING_XOR_DIAG_OUTPUT", "4"))
const WEIGHT_SEED = parse(Int, get(ENV, "ISING_XOR_DIAG_WEIGHT_SEED", "2"))
const BIAS_SEED = parse(Int, get(ENV, "ISING_XOR_DIAG_BIAS_SEED", "11"))
const BIAS_SCALE = parse(FT, get(ENV, "ISING_XOR_DIAG_BIAS_SCALE", "0.1"))
const WEIGHT_SCALE = parse(FT, get(ENV, "ISING_XOR_DIAG_WEIGHT_SCALE", "0.03"))
const WEIGHT_NORM = parse(FT, get(ENV, "ISING_XOR_DIAG_WEIGHT_NORM", "0.2"))
const STEPSIZE = parse(FT, get(ENV, "ISING_XOR_DIAG_STEPSIZE", "0.001"))
const BLOCK_SIZE = parse(Int, get(ENV, "ISING_XOR_DIAG_BLOCK_SIZE", "8"))
const INPUT_BIAS = parse(Bool, lowercase(get(ENV, "ISING_XOR_DIAG_INPUT_BIAS", "true")))
const BASE_SEED = parse(Int, get(ENV, "ISING_XOR_DIAG_BASE_SEED", "30000"))

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
    p_vertical = pattern_vertical()
    p_horizontal = pattern_horizontal()
    va = a ? p_vertical : -p_vertical
    hb = b ? p_horizontal : -p_horizontal
    x = zeros(FT, input_units())
    x[1:16] .= FT(0.5) .* (va .+ hb)
    INPUT_BIAS && (x[17] = one(FT))
    return x
end

xor_label(a::Bool, b::Bool) = xor(a, b)

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

function readout_vector()
    return (output_pattern(true) .- output_pattern(false)) ./ FT(OUTPUT_UNITS)
end

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
    return graph
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

function summarize(values)
    absvalues = abs.(values)
    return (;
        mean = mean(values),
        min = minimum(values),
        max = maximum(values),
        meanabs = mean(absvalues),
        frac_09 = count(>=(FT(0.9)), absvalues) / length(absvalues),
        frac_099 = count(>=(FT(0.99)), absvalues) / length(absvalues),
        frac_mid = count(<(FT(0.5)), absvalues) / length(absvalues),
    )
end

function print_summary(name, stats)
    @printf(
        "    %-6s mean=% .4f min=% .4f max=% .4f mean|s|=%.4f |s|>=.9=%.3f |s|>=.99=%.3f |s|<.5=%.3f\n",
        name,
        stats.mean,
        stats.min,
        stats.max,
        stats.meanabs,
        stats.frac_09,
        stats.frac_099,
        stats.frac_mid,
    )
end

function run_once!(graph, x, seed)
    dynamics = BlockLangevin(
        stepsize = STEPSIZE,
        adjusted = true,
        block_size = BLOCK_SIZE,
        group_steps = 1,
    )
    InteractiveIsing.temp!(graph, FT(1e-4))
    context = Processes.init(dynamics, (; model = graph))
    Random.seed!(context.rng, seed)
    resetstate!(context.model)
    InteractiveIsing.off!(context.model.index_set, 1)
    state(context.model[1]) .= x
    for _ in 1:RELAXATION_STEPS
        context = merge(context, Processes.step!(dynamics, context))
    end
    return copy(state(context.model[2])), copy(state(context.model[3]))
end

function main()
    graph = normalize_weights!(xor_graph())
    cases = ((false, false), (false, true), (true, false), (true, true))

    println("XOR relaxed-state diagnostics")
    println("  repeats=$REPEATS relaxation=$RELAXATION_STEPS hidden=$HIDDEN_UNITS output=$OUTPUT_UNITS weight_norm=$WEIGHT_NORM")

    for (case_idx, (a, b)) in enumerate(cases)
        hidden = Matrix{FT}(undef, HIDDEN_UNITS, REPEATS)
        output = Matrix{FT}(undef, OUTPUT_UNITS, REPEATS)
        scores = Vector{FT}(undef, REPEATS)
        x = xor_input(a, b)

        for repeat_idx in 1:REPEATS
            hidden[:, repeat_idx], output[:, repeat_idx] =
                run_once!(deepcopy(graph), x, BASE_SEED + 1000 * case_idx + repeat_idx)
            scores[repeat_idx] = readout_score(view(output, :, repeat_idx))
        end

        truth = xor_label(a, b)
        target = truth ? one(FT) : -one(FT)
        acc = count(==(truth), scores .> zero(FT)) / REPEATS
        mse = mean(abs2, scores .- target)

        println("  case=($a, $b) truth=$truth target=$target acc=$acc score_mse=$(round(mse, digits=6))")
        @printf(
            "    score mean=% .4f min=% .4f max=% .4f std=%.4f\n",
            mean(scores),
            minimum(scores),
            maximum(scores),
            std(scores),
        )
        print_summary("hidden", summarize(vec(hidden)))
        print_summary("output", summarize(vec(output)))
    end
end

main()
