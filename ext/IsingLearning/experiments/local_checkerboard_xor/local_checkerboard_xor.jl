using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using Optimisers
using Random
using SparseArrays
using LinearAlgebra
using Statistics
using Dates
using CairoMakie
using IsingLearning.InteractiveIsing.Processes

const FT = Float64
const II = IsingLearning.InteractiveIsing
const Processes = II.Processes

const DEFAULT_RUN_ROOT = joinpath(@__DIR__, "runs")
const CASES = ((false, false), (true, false), (false, true), (true, true))
const CASE_NAMES = ("00", "10", "01", "11")
const CHECKER_INTERNAL_RNGS = [Ref(Random.MersenneTwister(i)) for i in 1:3]
const CHECKER_INTERNAL_SCALES = [Ref(zero(FT)) for _ in 1:3]

checker_internal_weight_1(; dr, c1 = nothing, c2 = nothing, dc = nothing) =
    CHECKER_INTERNAL_SCALES[1][] * randn(CHECKER_INTERNAL_RNGS[1][], FT)
checker_internal_weight_2(; dr, c1 = nothing, c2 = nothing, dc = nothing) =
    CHECKER_INTERNAL_SCALES[2][] * randn(CHECKER_INTERNAL_RNGS[2][], FT)
checker_internal_weight_3(; dr, c1 = nothing, c2 = nothing, dc = nothing) =
    CHECKER_INTERNAL_SCALES[3][] * randn(CHECKER_INTERNAL_RNGS[3][], FT)
const CHECKER_INTERNAL_WEIGHT_FUNCS = (checker_internal_weight_1, checker_internal_weight_2, checker_internal_weight_3)

"""
    LocalCheckerboardConfig(; kwargs...)

Configuration for a local XOR experiment with three square layers.

The input bits are not encoded as four separate cases. Bit `a = 1` freezes the
white cells of checkerboard pattern A in the input layer to `+1`; bit `b = 1`
freezes the complementary checkerboard pattern B to `+1`. Thus `(0, 0)`
freezes nothing, `(1, 0)` freezes half the layer, `(0, 1)` freezes the other
half, and `(1, 1)` freezes the full input layer.
"""
Base.@kwdef struct LocalCheckerboardConfig
    name::String = "checker_2x2_global"
    side::Int = 2
    hidden_side::Int = side
    code_side::Int = side
    code_stride::Int = 1
    code_offset::Tuple{Int,Int} = (1, 1)
    epochs::Int = 1000
    log_every::Int = 100
    minit::Int = 4
    eval_repeats::Int = 16
    workers::Int = max(1, min(Threads.nthreads(), 8))
    free_relaxation::Int = 100
    nudged_relaxation::Int = 100
    β::FT = 0.05
    lr::FT = 0.005
    weight_decay::FT = 1e-4
    grad_clip::FT = 20.0
    temp::FT = 0.005
    temp_is_factor::Bool = false
    stepsize::FT = 0.05
    block_size::Int = 8
    inter_radius::FT = sqrt(2.0) + 1e-6
    internal_nn::Int = 1
    inter_weight_scale::FT = 0.05
    input_internal_scale::FT = 0.02
    hidden_internal_scale::FT = 0.02
    output_internal_scale::FT = 0.02
    bias_scale::FT = 0.02
    weight_seed::Int = 2
    internal_seed::Int = 3
    bias_seed::Int = 11
    base_seed::Int = 91000
    init_mode::Symbol = :random
    state_mode::Symbol = :continuous
    dynamics_mode::Symbol = :langevin
    output_clamp_mode::Symbol = :readout
    doublewell_barrier::FT = 0.0
    free_temp_start_factor::FT = 1.0
    free_temp_stop_factor::FT = 1.0
    nudged_temp_start_factor::FT = 1.0
    nudged_temp_stop_factor::FT = 1.0
    temp_schedule_power::FT = 1.0
end

"""
    StaticDoubleWell(barrier)

Experiment-local, non-trainable double-well potential

```math
V(s) = barrier * (s^4 - 2s^2)
```

up to an irrelevant constant. On `[-1, 1]` its minima are at `s = ±1`; the
energy barrier from either minimum to `s = 0` is `barrier`. This term is kept
out of `IsingLearning.contrastive_gradient`, so it stabilizes the states without
becoming a trainable local-potential parameter.
"""
struct StaticDoubleWell{T<:Real} <: II.LocalPotential
    barrier::T
end

@inline function II.calculate(::II.ΔH, hterm::StaticDoubleWell, model::II.AbstractIsingGraph, proposal)
    spins = II.graphstate(model)
    old = spins[II.at_idx(proposal)]
    new = II.to_val(proposal)
    α = hterm.barrier
    return α * (new^4 - 2new^2 - old^4 + 2old^2)
end

@inline function II.calculate(::II.d_iH, hterm::StaticDoubleWell, model::II.AbstractIsingGraph, s_idx)
    s = II.graphstate(model)[s_idx]
    α = hterm.barrier
    return 4α * (s^3 - s)
end

@inline function II.calculate(::II.H_i, hterm::StaticDoubleWell, model::II.AbstractIsingGraph, idx)
    s = II.graphstate(model)[idx]
    α = hterm.barrier
    return α * (s^4 - 2s^2)
end

@inline function II.calculate(::II.H, hterm::StaticDoubleWell, model::II.AbstractIsingGraph)
    total = zero(eltype(model))
    α = hterm.barrier
    @inbounds for s in II.graphstate(model)
        total += α * (s^4 - 2s^2)
    end
    return total
end

raw"""
    OutputPatternClamping(output_idxs; β = 0, target)

Experiment-local output-only squared-error clamping term:

```math
H_\mathrm{out}(s) = \frac{\beta}{2}\sum_k (s_{i_k} - y_k)^2
```

where `output_idxs = (i_1, i_2, ...)`. Unlike `InteractiveIsing.Clamping`,
this term has no force on non-output spins, so it can safely be used when the
target vector only describes a physical output code rather than the full graph
state.
"""
struct OutputPatternClamping{B,T,I} <: II.HamiltonianTerm
    β::B
    target::T
    output_idxs::I
end

raw"""
    MultiLinearReadoutClamping(output_idxs, readouts; β = 0, target)

Experiment-local squared-error clamping on multiple linear readouts of the
same physical output layer:

```math
H_\mathrm{out}(s) =
\frac{\beta}{2}\sum_c (r_c(s) - y_c)^2,
\quad r_c(s) = \sum_k W_{kc}s_{i_k}.
```

For checkerboard XOR this is used as a two-channel local readout: one channel
averages checkerboard mask A and the other averages checkerboard mask B.
"""
struct MultiLinearReadoutClamping{B,T,R,I} <: II.HamiltonianTerm
    β::B
    target::T
    readouts::R
    output_idxs::I
end

function MultiLinearReadoutClamping(output_idxs, readouts; β = 0, target)
    idxs = collect(Int, output_idxs)
    isempty(idxs) && throw(ArgumentError("output_idxs cannot be empty"))
    readout_mat = Matrix{FT}(readouts)
    size(readout_mat, 1) == length(idxs) ||
        throw(ArgumentError("readouts must have one row per output index"))
    target_vec = collect(FT, target)
    length(target_vec) == size(readout_mat, 2) ||
        throw(ArgumentError("target length must match number of readout columns"))
    return MultiLinearReadoutClamping(II.UniformArray(FT(β)), target_vec, readout_mat, idxs)
end

function multi_readout_scores(hterm::MultiLinearReadoutClamping, state)
    scores = zeros(FT, length(hterm.target))
    @inbounds for pos in eachindex(hterm.output_idxs)
        sval = FT(state[hterm.output_idxs[pos]])
        for channel in eachindex(scores)
            scores[channel] += hterm.readouts[pos, channel] * sval
        end
    end
    return scores
end

_multi_readout_position(hterm::MultiLinearReadoutClamping, state_idx::Integer) =
    findfirst(==(state_idx), hterm.output_idxs)

function II.calculate(::II.H, hterm::MultiLinearReadoutClamping, model::II.AbstractIsingGraph)
    scores = multi_readout_scores(hterm, II.graphstate(model))
    return hterm.β[] / 2 * sum(abs2, scores .- hterm.target)
end

function II.calculate(::II.d_iH, hterm::MultiLinearReadoutClamping, model::II.AbstractIsingGraph, s_idx)
    pos = _multi_readout_position(hterm, s_idx)
    isnothing(pos) && return zero(eltype(model))
    residuals = multi_readout_scores(hterm, II.graphstate(model)) .- hterm.target
    total = zero(FT)
    @inbounds for channel in eachindex(residuals)
        total += residuals[channel] * hterm.readouts[pos, channel]
    end
    return hterm.β[] * total
end

function II.calculate(::II.ΔH, hterm::MultiLinearReadoutClamping, model::II.AbstractIsingGraph, proposal::II.FlipProposal)
    pos = _multi_readout_position(hterm, II.at_idx(proposal))
    isnothing(pos) && return zero(eltype(model))
    old_scores = multi_readout_scores(hterm, II.graphstate(model))
    new_scores = copy(old_scores)
    δ = FT(II.to_val(proposal) - II.from_val(proposal))
    @inbounds for channel in eachindex(new_scores)
        new_scores[channel] += δ * hterm.readouts[pos, channel]
    end
    return hterm.β[] / 2 * (sum(abs2, new_scores .- hterm.target) - sum(abs2, old_scores .- hterm.target))
end

function II.calculate(::II.ΔH, hterm::MultiLinearReadoutClamping, model::II.AbstractIsingGraph, proposal::II.MultiSpinProposal)
    old_scores = multi_readout_scores(hterm, II.graphstate(model))
    new_scores = copy(old_scores)
    @inbounds for proposal_idx in 1:length(proposal)
        pos = _multi_readout_position(hterm, II.at_idx(proposal, proposal_idx))
        isnothing(pos) && continue
        δ = FT(II.to_val(proposal, proposal_idx) - II.from_val(proposal, proposal_idx))
        for channel in eachindex(new_scores)
            new_scores[channel] += δ * hterm.readouts[pos, channel]
        end
    end
    return hterm.β[] / 2 * (sum(abs2, new_scores .- hterm.target) - sum(abs2, old_scores .- hterm.target))
end

function OutputPatternClamping(output_idxs; β = 0, target)
    idxs = collect(Int, output_idxs)
    isempty(idxs) && throw(ArgumentError("output_idxs cannot be empty"))
    target_vec = collect(FT, target)
    length(target_vec) == length(idxs) ||
        throw(ArgumentError("target length $(length(target_vec)) must match output_idxs length $(length(idxs))"))
    return OutputPatternClamping(II.UniformArray(FT(β)), target_vec, idxs)
end

function _output_pattern_position(hterm::OutputPatternClamping, state_idx::Integer)
    return findfirst(==(state_idx), hterm.output_idxs)
end

function II.calculate(::II.H, hterm::OutputPatternClamping, model::II.AbstractIsingGraph)
    state = II.graphstate(model)
    total = zero(FT)
    @inbounds for k in eachindex(hterm.output_idxs)
        residual = FT(state[hterm.output_idxs[k]]) - hterm.target[k]
        total += residual^2
    end
    return hterm.β[] / 2 * total
end

function II.calculate(::II.d_iH, hterm::OutputPatternClamping, model::II.AbstractIsingGraph, s_idx)
    pos = _output_pattern_position(hterm, s_idx)
    isnothing(pos) && return zero(eltype(model))
    return hterm.β[] * (II.graphstate(model)[s_idx] - hterm.target[pos])
end

function II.calculate(::II.ΔH, hterm::OutputPatternClamping, model::II.AbstractIsingGraph, proposal::II.FlipProposal)
    pos = _output_pattern_position(hterm, II.at_idx(proposal))
    isnothing(pos) && return zero(eltype(model))
    old = II.from_val(proposal)
    new = II.to_val(proposal)
    target = hterm.target[pos]
    return hterm.β[] / 2 * ((new - target)^2 - (old - target)^2)
end

function II.calculate(::II.ΔH, hterm::OutputPatternClamping, model::II.AbstractIsingGraph, proposal::II.MultiSpinProposal)
    total = zero(FT)
    @inbounds for proposal_idx in 1:length(proposal)
        pos = _output_pattern_position(hterm, II.at_idx(proposal, proposal_idx))
        isnothing(pos) && continue
        old = II.from_val(proposal, proposal_idx)
        new = II.to_val(proposal, proposal_idx)
        target = hterm.target[pos]
        total += (new - target)^2 - (old - target)^2
    end
    return hterm.β[] / 2 * total
end

doublewell_barrier(config::LocalCheckerboardConfig) = max(zero(FT), FT(config.doublewell_barrier))

function _scheduled_temperature(base_temp::Real, barrier::Real, start_factor::Real, stop_factor::Real)
    scale = max(FT(base_temp), FT(barrier), eps(FT))
    return FT(start_factor) * scale, FT(stop_factor) * FT(base_temp)
end

function free_temperature_schedule(config::LocalCheckerboardConfig, graph)
    base = effective_temp(graph, config)
    return _scheduled_temperature(base, doublewell_barrier(config), config.free_temp_start_factor, config.free_temp_stop_factor)
end

function nudged_temperature_schedule(config::LocalCheckerboardConfig, graph)
    base = effective_temp(graph, config)
    return _scheduled_temperature(base, doublewell_barrier(config), config.nudged_temp_start_factor, config.nudged_temp_stop_factor)
end

"""
    TemperatureAnnealedSampler(sampler, start_T, stop_T, power, n_steps)

Experiment-local sampler wrapper. Each `step!` writes a power-law scheduled
temperature to `context.model`, then delegates to the wrapped sampler. The
wrapped sampler is otherwise unchanged, so this can be used with BlockLangevin,
GlobalLangevin, or Metropolis without changing their implementations.
"""
struct TemperatureAnnealedSampler{Name,A,T<:Real} <: II.IsingMCAlgorithm
    sampler::A
    start_T::T
    stop_T::T
    power::T
    n_steps::Int
end

function TemperatureAnnealedSampler(::Val{Name}, sampler, start_T::Real, stop_T::Real, power::Real, n_steps::Integer) where {Name}
    start_T, stop_T, power = promote(FT(start_T), FT(stop_T), FT(power))
    return TemperatureAnnealedSampler{Name,typeof(sampler),typeof(start_T)}(sampler, start_T, stop_T, power, max(1, Int(n_steps)))
end

function Processes.init(algorithm::TemperatureAnnealedSampler, context)
    inner = Processes.init(algorithm.sampler, context)
    return merge(inner, (; anneal_step = Ref(0)))
end

function Processes.step!(algorithm::TemperatureAnnealedSampler, context)
    step_idx = context.anneal_step[]
    total = max(algorithm.n_steps, 1)
    progress = total == 1 ? one(FT) : FT(step_idx) / FT(total - 1)
    Tcur = algorithm.stop_T + (algorithm.start_T - algorithm.stop_T) * (one(FT) - clamp(progress, zero(FT), one(FT)))^algorithm.power
    II.temp!(context.model, Tcur)
    out = Processes.step!(algorithm.sampler, context)
    context.anneal_step[] = min(step_idx + 1, total - 1)
    return merge(out, (; anneal_T = Tcur))
end

"""
    max_local_interaction_energy(g)

Return a conservative local energy scale for temperature tuning. For a bounded
spin `s_i in [-1, 1]`, the largest interaction-driven flip energy is bounded by
`2 * sum_j |J_ij|`. The maximum of that value over spins is a useful scale for
Metropolis acceptance and for Langevin noise strength.
"""
function max_local_interaction_energy(g)
    J = g.adj.sp
    fields = vec(sum(abs.(J); dims = 2))
    isempty(fields) && return one(FT)
    return max(FT(2) * maximum(fields), eps(FT))
end

effective_temp(g, config::LocalCheckerboardConfig) =
    config.temp_is_factor ? FT(config.temp) * max_local_interaction_energy(g) : FT(config.temp)

"""
    CheckerboardInputIndexSet(g, frozen_sets)

Mutable index set whose active list is selected from precomputed input-bit
freeze masks. `frozen_sets[k]` contains the graph indices frozen for the `k`th
XOR case in `CASES`. Sampling excludes those frozen input sites and includes all
other graph sites, so non-code input spins remain dynamical for embedded codes.
"""
struct CheckerboardInputIndexSet{I} <: II.UniformIndexPicker
    all_active::Vector{I}
    active_by_case::NTuple{4,Vector{I}}
    active::Base.RefValue{Vector{I}}
    case_idx::Base.RefValue{Int}
    changed::Base.RefValue{Bool}
end

function CheckerboardInputIndexSet(g, frozen_sets::NTuple{4,Vector{Int}})
    all_active = collect(Int, II.graphidxs(g))
    active_by_case = ntuple(4) do case_idx
        frozen = Set(frozen_sets[case_idx])
        [idx for idx in all_active if !(idx in frozen)]
    end
    return CheckerboardInputIndexSet(all_active, active_by_case, Ref(active_by_case[1]), Ref(1), Ref(true))
end

@inline II.sampling_indices(is::CheckerboardInputIndexSet) = is.active[]
@inline II.consume_changed!(is::CheckerboardInputIndexSet) = (v = is.changed[]; is.changed[] = false; v)
@inline II.pick_idx(rng::Random.AbstractRNG, is::CheckerboardInputIndexSet) = rand(rng, is.active[])

"""
    set_input_case!(index_set, case_idx)

Switch the active sampler set to the precomputed mask for one XOR input case.
"""
function set_input_case!(is::CheckerboardInputIndexSet, case_idx::Integer)
    1 <= case_idx <= 4 || throw(ArgumentError("case_idx must be in 1:4, got $case_idx"))
    if is.case_idx[] != case_idx
        is.case_idx[] = Int(case_idx)
        is.active[] = is.active_by_case[Int(case_idx)]
        is.changed[] = true
    end
    return is
end

"""
    checker_code_positions(side, code_side, stride, offset)

Return local `(row, col)` positions for a square checkerboard code embedded in a
larger square layer. With `stride = 1` and `code_side = side` this is a global
code. With `stride > 1`, unused layer spins lie between code sites and are never
directly frozen by the input bits.
"""
function checker_code_positions(side::Integer, code_side::Integer, stride::Integer, offset::Tuple{Int,Int})
    side > 0 || throw(ArgumentError("side must be positive"))
    code_side > 0 || throw(ArgumentError("code_side must be positive"))
    stride > 0 || throw(ArgumentError("stride must be positive"))
    max_r = offset[1] + (code_side - 1) * stride
    max_c = offset[2] + (code_side - 1) * stride
    max(max_r, max_c) <= side ||
        throw(ArgumentError("embedded code does not fit: side=$side code_side=$code_side stride=$stride offset=$offset"))
    return [(r, c) for r in offset[1]:stride:max_r for c in offset[2]:stride:max_c]
end

"""
    checker_masks(positions)

Split embedded code positions into complementary checkerboard masks A and B.
Mask A is the white parity `(row + col) is even`; mask B is the inverted mask.
"""
function checker_masks(positions)
    a = Tuple{Int,Int}[]
    b = Tuple{Int,Int}[]
    for pos in positions
        (isodd(sum(pos)) ? b : a) |> x -> push!(x, pos)
    end
    return (; a, b)
end

"""
    global_idxs_for_positions(layer, positions)

Map local two-dimensional layer positions to global graph indices.
"""
function global_idxs_for_positions(layer, positions)
    idxs = collect(II.graphidxs(layer))
    dims = size(II.state(layer))
    return [idxs[LinearIndices(dims)[pos...]] for pos in positions]
end

"""
    readout_vector(output_layer, code_positions)

Build the scalar XOR readout weights on the output layer. The readout is `+1`
on checkerboard B and `-1` on checkerboard A, normalized by the number of code
sites. The target is therefore `-1` for XOR false and `+1` for XOR true.
"""
function readout_vector(output_layer, code_positions)
    output_idxs = global_idxs_for_positions(output_layer, code_positions)
    masks = checker_masks(code_positions)
    a_lookup = Set(masks.a)
    norm = FT(length(code_positions))
    weights = [pos in a_lookup ? -one(FT) / norm : one(FT) / norm for pos in code_positions]
    return output_idxs, weights
end

function two_channel_readout_matrix(code_positions)
    masks = checker_masks(code_positions)
    a_lookup = Set(masks.a)
    n_a = max(1, length(masks.a))
    n_b = max(1, length(masks.b))
    readouts = zeros(FT, length(code_positions), 2)
    for (idx, pos) in enumerate(code_positions)
        if pos in a_lookup
            readouts[idx, 1] = inv(FT(n_a))
        else
            readouts[idx, 2] = inv(FT(n_b))
        end
    end
    return readouts
end

two_channel_target(target::Real) = target < 0 ? FT[1, -1] : FT[-1, 1]

function checker_output_target_vector(config::LocalCheckerboardConfig, target::Real)
    positions = checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)
    masks = checker_masks(positions)
    b_lookup = Set(masks.b)
    return [pos in b_lookup ? FT(target) : -FT(target) for pos in positions]
end

target_dim(config::LocalCheckerboardConfig) =
    config.output_clamp_mode === :readout ? 1 :
    config.output_clamp_mode === :two_readout ? 2 :
    config.output_clamp_mode === :pattern ? length(checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)) :
    throw(ArgumentError("output_clamp_mode must be :readout, :two_readout, or :pattern"))

case_index(a::Bool, b::Bool) = (a ? 2 : 0) + (b ? 1 : 0) + 1
xor_target(a::Bool, b::Bool) = xor(a, b) ? one(FT) : -one(FT)

"""
    input_frozen_sets(g, config)

Precompute all four frozen input index sets from the two checkerboard input
bits. This is intentionally done once at graph construction time instead of
recomputing membership on every input application.
"""
function input_frozen_sets(g, config::LocalCheckerboardConfig)
    positions = checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)
    masks = checker_masks(positions)
    a_idxs = global_idxs_for_positions(g[1], masks.a)
    b_idxs = global_idxs_for_positions(g[1], masks.b)
    return ntuple(4) do idx
        a, b = CASES[idx]
        frozen = Int[]
        a && append!(frozen, a_idxs)
        b && append!(frozen, b_idxs)
        unique!(frozen)
        frozen
    end
end

function case_from_input(x)
    length(x) == 2 || throw(ArgumentError("checkerboard XOR input must have length 2, got $(length(x))"))
    return case_index(x[1] > 0, x[2] > 0)
end

"""
    apply_checker_input!(g, x, config)

Apply one two-bit XOR input to `g`: choose the precomputed active index set and
write `+1` to the frozen input checkerboard sites. Sites belonging to inactive
bits are not clamped and are not overwritten here.
"""
function apply_checker_input!(g, x, config::LocalCheckerboardConfig)
    case_idx = case_from_input(x)
    set_input_case!(g.index_set, case_idx)
    frozen = get(g, :checker_frozen_sets, nothing)
    isnothing(frozen) && error("graph is missing :checker_frozen_sets addon")
    s = II.state(g)
    @inbounds for idx in frozen[case_idx]
        s[idx] = one(eltype(s))
    end
    return g
end

function signed_interlayer_weight_generator(config::LocalCheckerboardConfig)
    rng = Random.MersenneTwister(config.weight_seed)
    return II.AllToAllWeightGenerator(
        (; dr, c1, c2, dc) -> dr <= config.inter_radius ? config.inter_weight_scale * randn(rng, FT) : zero(FT),
        rng,
    )
end

function internal_weight_generator(slot::Integer, seed::Integer, nn::Integer, scale::Real)
    1 <= slot <= 3 || throw(ArgumentError("internal weight-generator slot must be in 1:3, got $slot"))
    CHECKER_INTERNAL_RNGS[slot][] = Random.MersenneTwister(seed)
    CHECKER_INTERNAL_SCALES[slot][] = FT(scale)
    return II.WeightGenerator(
        CHECKER_INTERNAL_WEIGHT_FUNCS[slot],
        nn,
        CHECKER_INTERNAL_RNGS[slot][];
        symmetric = true,
    )
end

function bias_generator(config::LocalCheckerboardConfig)
    rng = Random.MersenneTwister(config.bias_seed)
    return g -> config.bias_scale .* randn(rng, FT, II.statelen(g))
end

"""
    symmetrize_adjacency!(g)

Force the graph adjacency to satisfy `J_ij == J_ji` by averaging every
off-diagonal pair in place. This is retained as a diagnostic/repair helper for
old runs. Current local checkerboard experiments use true symmetric in-layer
weight generation and should already satisfy this check without repair.

The mutation is deliberately in-place on `adj(g).sp`, because the instantiated
`Bilinear` Hamiltonian stores the same adjacency object.
"""
function symmetrize_adjacency!(g)
    A = II.adj(g).sp
    rows, cols, _ = SparseArrays.findnz(A)
    for k in eachindex(rows)
        i = rows[k]
        j = cols[k]
        i < j || continue
        avg = (A[i, j] + A[j, i]) / FT(2)
        A[i, j] = avg
        A[j, i] = avg
    end
    return g
end

"""
    adjacency_symmetry_error(g)

Return the largest absolute off-diagonal mismatch `|J_ij - J_ji|`.
"""
function adjacency_symmetry_error(g)
    A = II.adj(g).sp
    rows, cols, _ = SparseArrays.findnz(A)
    err = zero(FT)
    for k in eachindex(rows)
        i = rows[k]
        j = cols[k]
        i == j && continue
        err = max(err, abs(FT(A[i, j] - A[j, i])))
    end
    return err
end

"""
    assert_symmetric_adjacency(g; atol = 1e-10)

Error if the graph adjacency is not symmetric to numerical tolerance.
"""
function assert_symmetric_adjacency(g; atol = FT(1e-10))
    err = adjacency_symmetry_error(g)
    err <= atol || error("adjacency is not symmetric: max |J_ij - J_ji| = $err")
    return g
end

"""
    checkerboard_graph(config)

Construct a three-layer local XOR graph:

`input(side x side) -> hidden(hidden_side x hidden_side) -> output(side x side)`.

All three layers can have internal neighborhood connections. Adjacent layers are
connected by a radius-limited signed random generator. The output loss is a
scalar `LinearReadoutClamping` on the output checkerboard code.
"""
function checkerboard_graph(config::LocalCheckerboardConfig)
    state_type =
        config.state_mode === :continuous ? II.Continuous() :
        config.state_mode === :discrete ? II.Discrete() :
        throw(ArgumentError("state_mode must be :continuous or :discrete"))

    layers = (
        II.Layer(
            config.side, config.side, II.StateSet(-one(FT), one(FT)),
            internal_weight_generator(1, config.internal_seed, config.internal_nn, config.input_internal_scale),
            state_type, II.Coords(0, 1, 0); periodic = false,
        ),
        II.Layer(
            config.hidden_side, config.hidden_side, II.StateSet(-one(FT), one(FT)),
            internal_weight_generator(2, config.internal_seed + 1, config.internal_nn, config.hidden_internal_scale),
            state_type, II.Coords(0, 2, 0); periodic = false,
        ),
        II.Layer(
            config.side, config.side, II.StateSet(-one(FT), one(FT)),
            internal_weight_generator(3, config.internal_seed + 2, config.internal_nn, config.output_internal_scale),
            state_type, II.Coords(0, 3, 0); periodic = false,
        ),
    )

    output_positions = checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)
    output_start = config.side^2 + config.hidden_side^2 + 1
    output_linear = LinearIndices((config.side, config.side))
    output_idxs = [output_start + output_linear[pos...] - 1 for pos in output_positions]
    masks = checker_masks(output_positions)
    a_lookup = Set(masks.a)
    readout = [pos in a_lookup ? -one(FT) / FT(length(output_positions)) : one(FT) / FT(length(output_positions)) for pos in output_positions]
    clamping =
        config.output_clamp_mode === :readout ? IsingLearning.LinearReadoutClamping(output_idxs, readout; β = zero(FT), target = zero(FT)) :
        config.output_clamp_mode === :two_readout ? MultiLinearReadoutClamping(output_idxs, two_channel_readout_matrix(output_positions); β = zero(FT), target = zeros(FT, 2)) :
        config.output_clamp_mode === :pattern ? OutputPatternClamping(output_idxs; β = zero(FT), target = zeros(FT, length(output_idxs))) :
        throw(ArgumentError("output_clamp_mode must be :readout, :two_readout, or :pattern"))

    hamiltonian = II.Bilinear() + II.MagField(b = bias_generator(config)) + clamping
    if doublewell_barrier(config) > zero(FT)
        hamiltonian = hamiltonian + StaticDoubleWell(doublewell_barrier(config))
    end
    inter = signed_interlayer_weight_generator(config)
    graph = II.IsingGraph(
        layers[1], deepcopy(inter), layers[2], deepcopy(inter), layers[3],
        hamiltonian;
        precision = FT,
        index_set = g -> CheckerboardInputIndexSet(g, input_frozen_sets(g, config)),
    )
    graph.addons[:checker_config] = config
    graph.addons[:checker_frozen_sets] = input_frozen_sets(graph, config)
    graph.addons[:checker_output_idxs] = output_idxs
    graph.addons[:checker_readout] = readout
    graph.addons[:checker_two_readout] = two_channel_readout_matrix(output_positions)
    assert_symmetric_adjacency(graph)
    II.temp!(graph, effective_temp(graph, config))
    return graph
end

function checkerboard_layer(graph, config::LocalCheckerboardConfig)
    dynamics =
        config.dynamics_mode === :langevin ? II.BlockLangevin(stepsize = config.stepsize, adjusted = false, block_size = config.block_size, group_steps = 1) :
        config.dynamics_mode === :local_langevin ? II.LocalLangevin(stepsize = config.stepsize, adjusted = false, order = :random, group_steps = 1) :
        config.dynamics_mode === :global_langevin ? II.GlobalLangevin(stepsize = config.stepsize, adjusted = false, group_steps = 1) :
        config.dynamics_mode === :metropolis ? II.IsingMetropolis() :
        throw(ArgumentError("dynamics_mode must be :langevin, :local_langevin, :global_langevin, or :metropolis"))

    return IsingLearning.LayeredIsingGraphLayer(
        () -> checkerboard_graph(config);
        input_idxs = 1:2,
        output_idxs = 1:target_dim(config),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = config.free_relaxation,
        free_relaxation_steps = config.free_relaxation,
        nudged_relaxation_steps = config.nudged_relaxation,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""
    checker_initstate!(g, config, rng)

Reset a worker graph before the free phase. `:random` samples valid states from
the graph's layer state sets. `:zero` is only allowed for continuous layers.
"""
function checker_initstate!(g, config::LocalCheckerboardConfig)
    s = II.state(g)
    if config.init_mode === :zero
        any(layer -> II.statetype(layer) isa II.Discrete, II.layers(g)) &&
            throw(ArgumentError("init_mode=:zero is invalid for discrete local checkerboard graphs; use init_mode=:random"))
        fill!(s, zero(eltype(s)))
    elseif config.init_mode === :random
        s .= II.initRandomState(g)
    else
        throw(ArgumentError("init_mode must be :zero or :random"))
    end
    set_input_case!(g.index_set, 1)
    return g
end

function CheckerForwardDynamics(layer, config::LocalCheckerboardConfig; dynamics_algorithm = layer.dynamics_algorithm)
    relaxation_steps = layer.free_relaxation_steps
    n_units = layer.nunits
    start_T, stop_T = free_temperature_schedule(config, layer.model_graph)
    dynamics_algorithm = TemperatureAnnealedSampler(Val(:free), deepcopy(dynamics_algorithm), start_T, stop_T, config.temp_schedule_power, relaxation_steps)

    forward = @Routine begin
        @alias free_dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        checker_initstate!(free_dynamics.model, config)
        apply_checker_input!(free_dynamics.model, x, config)
        model = @repeat relaxation_steps free_dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(m -> II.state(m), model))
    end
    return (; algorithm = forward, dynamics = forward.free_dynamics)
end

function checker_find_hamiltonian_term(hts, ::Type{T}) where {T}
    for hterm in II.hamiltonians(hts)
        hterm isa T && return hterm
    end
    return nothing
end

function checker_learning_clamping(g)
    readout = checker_find_hamiltonian_term(g.hamiltonian, IsingLearning.LinearReadoutClamping)
    !isnothing(readout) && return readout

    multi = checker_find_hamiltonian_term(g.hamiltonian, MultiLinearReadoutClamping)
    !isnothing(multi) && return multi

    pattern = checker_find_hamiltonian_term(g.hamiltonian, OutputPatternClamping)
    !isnothing(pattern) && return pattern

    error("checker graph has no supported checker clamping term")
end

function checker_apply_targets!(g, y)
    clamping = checker_learning_clamping(g)
    if clamping isa IsingLearning.LinearReadoutClamping
        isempty(y) && throw(ArgumentError("LinearReadoutClamping needs a scalar target in y[1]"))
        clamping.target[] = first(y)
    elseif clamping isa MultiLinearReadoutClamping
        length(y) == length(clamping.target) ||
            throw(ArgumentError("MultiLinearReadoutClamping target length $(length(y)) must match $(length(clamping.target))"))
        clamping.target .= y
    elseif clamping isa OutputPatternClamping
        length(y) == length(clamping.target) ||
            throw(ArgumentError("OutputPatternClamping target length $(length(y)) must match $(length(clamping.target))"))
        clamping.target .= y
    else
        error("unsupported checker clamping term $(typeof(clamping))")
    end
    return g
end

function checker_set_clamping_beta!(g, β)
    checker_learning_clamping(g).β[] = FT(β)
    return g
end

function CheckerNudgedDynamics(layer, config::LocalCheckerboardConfig)
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    start_T, stop_T = nudged_temperature_schedule(config, layer.model_graph)
    plus_dynamics_algorithm = TemperatureAnnealedSampler(Val(:plus), deepcopy(layer.nudged_dynamics_algorithm), start_T, stop_T, config.temp_schedule_power, relaxation_steps)
    minus_dynamics_algorithm = TemperatureAnnealedSampler(Val(:minus), deepcopy(layer.nudged_dynamics_algorithm), start_T, stop_T, config.temp_schedule_power, relaxation_steps)

    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias plus_dynamics = plus_dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = plus_dynamics.model, target = equilibrium_state)
        apply_checker_input!(plus_dynamics.model, x, config)
        checker_apply_targets!(plus_dynamics.model, y)
        checker_set_clamping_beta!(plus_dynamics.model, beta)
        model = @repeat relaxation_steps plus_dynamics()
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias minus_dynamics = minus_dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = minus_dynamics.model, target = equilibrium_state)
        apply_checker_input!(minus_dynamics.model, x, config)
        checker_apply_targets!(minus_dynamics.model, y)
        checker_set_clamping_beta!(minus_dynamics.model, -beta)
        model = @repeat relaxation_steps minus_dynamics()
        minus_capture(isinggraph = model)
    end

    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = plus()
        @context c2 = minus()
    end
    return (; algorithm = final, plus_capture, minus_capture, dynamics = plus.plus_dynamics)
end

function CheckerForwardAndNudged(layer, config::LocalCheckerboardConfig)
    forward = CheckerForwardDynamics(layer, config).algorithm
    nudged = CheckerNudgedDynamics(layer, config)
    beta = layer.β
    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = forward()
        @context c2 = nudged.algorithm()
        checker_set_clamping_beta!(c1.free_dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(c1.free_dynamics.model, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers)
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.free_dynamics)
end

function checker_worker_process(layer, graph, config::LocalCheckerboardConfig)
    algo = Processes.resolve(CheckerForwardAndNudged(layer, config).algorithm)
    buffers = IsingLearning.gradient_buffer(graph)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            y = zeros(FT, target_dim(config)),
            buffers = buffers,
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:free_dynamics, graph, config.base_seed),
        dynamics_input(:plus_dynamics, graph, config.base_seed + 10_000),
        dynamics_input(:minus_dynamics, graph, config.base_seed + 20_000),
        Init(:plus_capture, state = graph),
        Init(:minus_capture, state = graph);
        repeat = 1,
    )
end

function checker_validation_process(layer, graph, config::LocalCheckerboardConfig)
    algo = Processes.resolve(CheckerForwardDynamics(layer, config; dynamics_algorithm = layer.validation_algorithm).algorithm)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:free_dynamics, graph, config.base_seed + 50_000);
        repeat = 1,
    )
end

mutable struct CheckerTrainer{L,G,P,S,W<:Process,V<:Process,O}
    layer::L
    prototype_graph::G
    params::P
    opt_state::S
    worker_graphs::Vector{G}
    workers::Vector{W}
    validation_graph::G
    validation_worker::V
    optimiser::O
end

function init_checker_trainer(layer, config::LocalCheckerboardConfig; graph = layer.model_graph, optimiser = Optimisers.Adam(config.lr))
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    workers = Process[]
    worker_graphs = typeof(graph)[]
    for _ in 1:config.workers
        wg = IsingLearning._worker_graph(graph, params)
        II.temp!(wg, effective_temp(wg, config))
        push!(worker_graphs, wg)
        push!(workers, checker_worker_process(layer, wg, config))
    end
    validation_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(validation_graph, effective_temp(validation_graph, config))
    validation_worker = checker_validation_process(layer, validation_graph, config)
    return CheckerTrainer(layer, graph, params, opt_state, worker_graphs, workers, validation_graph, validation_worker, optimiser)
end

function close_checker_trainer!(trainer::CheckerTrainer)
    for worker in trainer.workers
        isnothing(worker.task) || close(worker)
    end
    isnothing(trainer.validation_worker.task) || close(trainer.validation_worker)
    return trainer
end

function _broadcast_params!(trainer::CheckerTrainer)
    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    foreach(g -> IsingLearning.sync_graph_params!(g, trainer.params), trainer.worker_graphs)
    IsingLearning.sync_graph_params!(trainer.validation_graph, trainer.params)
    return trainer
end

function xor_inputs_targets(config::LocalCheckerboardConfig)
    x = zeros(FT, 2, 4)
    y = zeros(FT, target_dim(config), 4)
    for (col, (a, b)) in enumerate(CASES)
        x[:, col] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        target = xor_target(a, b)
        if config.output_clamp_mode === :readout
            y[1, col] = target
        elseif config.output_clamp_mode === :two_readout
            y[:, col] .= two_channel_target(target)
        elseif config.output_clamp_mode === :pattern
            y[:, col] .= checker_output_target_vector(config, target)
        else
            throw(ArgumentError("output_clamp_mode must be :readout, :two_readout, or :pattern"))
        end
    end
    return x, y
end

function seed_sampler_rng!(worker, seed::Integer)
    for (offset, name) in enumerate((:dynamics, :free_dynamics, :plus_dynamics, :minus_dynamics))
        hasproperty(Processes.context(worker), name) || continue
        dynamics_context = getproperty(Processes.context(worker), name)
        rng = hasproperty(dynamics_context, :rng) ? getproperty(dynamics_context, :rng) : nothing
        isnothing(rng) || Random.seed!(rng, seed + 10_000 * offset)
    end
    return worker
end

function dynamics_input(name::Symbol, graph, seed::Integer)
    return Init(name, model = graph, rng = Random.MersenneTwister(seed))
end

function start_worker!(worker, x, y; seed)
    Random.seed!(seed)
    context = Processes.context(worker)
    IsingLearning.zero_buffer!(context._state.buffers)
    context._state.x .= x
    context._state.y .= y
    Processes.reset!(worker)
    seed_sampler_rng!(worker, seed)
    run(worker)
    return worker
end

function finish_worker!(worker)
    wait(worker)
    close(worker)
    return worker
end

function finish_worker!(worker, batch_gradient, responses)
    finish_worker!(worker)
    free_state = Processes.context(worker)._state.equilibrium_state
    plus_state = Processes.context(worker).plus_capture.captured
    minus_state = Processes.context(worker).minus_capture.captured
    push!(responses, (sqrt(mean(abs2, plus_state .- free_state)) + sqrt(mean(abs2, minus_state .- free_state))) / FT(2))
    IsingLearning.add_buffer!(batch_gradient, Processes.context(worker)._state.buffers)
    return worker
end

function run_validation!(worker, x; seed)
    Random.seed!(seed)
    context = Processes.context(worker)
    context._state.x .= x
    Processes.reset!(worker)
    seed_sampler_rng!(worker, seed)
    run(worker)
    wait(worker)
    close(worker)
    return Processes.context(worker)._state.equilibrium_state
end

function add_weight_decay!(gradient, params, λ::Real)
    iszero(λ) && return gradient
    gradient.w .+= FT(λ) .* params.w
    gradient.b .+= FT(λ) .* params.b
    return gradient
end

function clip_gradient!(gradient, maxnorm::Real)
    isfinite(maxnorm) || return gradient
    normv = sqrt(sum(abs2, gradient.w) + sum(abs2, gradient.b))
    if normv > maxnorm
        scale = FT(maxnorm) / FT(normv)
        gradient.w .*= scale
        gradient.b .*= scale
    end
    return gradient
end

function train_epoch!(trainer::CheckerTrainer, x, y, batch_gradient, epoch::Integer, config::LocalCheckerboardConfig)
    IsingLearning.zero_buffer!(batch_gradient)
    nsamples = size(x, 2)
    responses = FT[]
    task_batch = Process[]
    seed_base = config.base_seed + epoch * 100_000
    job = 0
    for sample_idx in 1:nsamples
        for init_idx in 1:config.minit
            job += 1
            worker = trainer.workers[mod1(job, length(trainer.workers))]
            start_worker!(worker, @view(x[:, sample_idx]), @view(y[:, sample_idx]); seed = seed_base + 257 * sample_idx + init_idx)
            push!(task_batch, worker)
            if length(task_batch) == length(trainer.workers)
                for task_worker in task_batch
                    finish_worker!(task_worker, batch_gradient, responses)
                end
                empty!(task_batch)
            end
        end
    end
    for task_worker in task_batch
        finish_worker!(task_worker, batch_gradient, responses)
    end
    scale = inv(FT(2) * FT(config.β) * FT(nsamples * config.minit))
    IsingLearning.scale_buffer!(batch_gradient, scale)
    add_weight_decay!(batch_gradient, trainer.params, config.weight_decay)
    clip_gradient!(batch_gradient, config.grad_clip)

    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    _broadcast_params!(trainer)
    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    return (; grad_norm, response_norm = mean(responses))
end

function readout_score(g, equilibrium_state)
    idxs = get(g, :checker_output_idxs, nothing)
    weights = get(g, :checker_readout, nothing)
    isnothing(idxs) && error("graph is missing :checker_output_idxs addon")
    return dot(weights, @view equilibrium_state[idxs])
end

function two_readout_scores(g, equilibrium_state)
    idxs = get(g, :checker_output_idxs, nothing)
    readouts = get(g, :checker_two_readout, nothing)
    isnothing(idxs) && error("graph is missing :checker_output_idxs addon")
    isnothing(readouts) && error("graph is missing :checker_two_readout addon")
    return vec(transpose(view(equilibrium_state, idxs)) * readouts)
end

function evaluate_checker!(trainer::CheckerTrainer, x, y, config::LocalCheckerboardConfig; seed_offset::Integer)
    nsamples = size(x, 2)
    if config.output_clamp_mode === :two_readout
        scores = zeros(FT, nsamples, config.eval_repeats, 2)
        for sample_idx in 1:nsamples
            for rep in 1:config.eval_repeats
                st = run_validation!(trainer.validation_worker, @view(x[:, sample_idx]); seed = seed_offset + 997 * sample_idx + rep)
                scores[sample_idx, rep, :] .= two_readout_scores(trainer.validation_graph, st)
            end
        end
        means2 = dropdims(mean(scores; dims = 2), dims = 2)
        stds2 = dropdims(std(scores; dims = 2), dims = 2)
        targets2 = zeros(FT, nsamples, 2)
        for (sample_idx, (a, b)) in enumerate(CASES)
            targets2[sample_idx, :] .= two_channel_target(xor_target(a, b))
        end
        mse = mean((means2 .- targets2) .^ 2)
        predicted = [means2[i, 2] > means2[i, 1] for i in 1:nsamples]
        target_positive = [xor_target(a, b) > 0 for (a, b) in CASES]
        decision = means2[:, 2] .- means2[:, 1]
        decision_std = sqrt.(stds2[:, 1].^2 .+ stds2[:, 2].^2)
        accuracy = mean(predicted .== target_positive)
        return (; mse, accuracy, means = decision, stds = decision_std, targets = [xor_target(a, b) for (a, b) in CASES], margins = abs.(decision), min_margin = minimum(abs.(decision)), means2, stds2, targets2)
    end

    scores = zeros(FT, nsamples, config.eval_repeats)
    for sample_idx in 1:nsamples
        for rep in 1:config.eval_repeats
            st = run_validation!(trainer.validation_worker, @view(x[:, sample_idx]); seed = seed_offset + 997 * sample_idx + rep)
            scores[sample_idx, rep] = readout_score(trainer.validation_graph, st)
        end
    end
    means = vec(mean(scores; dims = 2))
    stds = vec(std(scores; dims = 2))
    targets = [xor_target(a, b) for (a, b) in CASES]
    mse = mean((means .- targets) .^ 2)
    accuracy = mean((means .> 0) .== (targets .> 0))
    margins = abs.(means)
    return (; mse, accuracy, means, stds, targets, margins, min_margin = minimum(margins))
end

function metric_row(config_name, epoch, metrics, grad_metrics, trainer::CheckerTrainer, initial_params)
    return Dict{String,Any}(
        "config" => config_name,
        "epoch" => epoch,
        "mse" => metrics.mse,
        "accuracy" => metrics.accuracy,
        "min_margin" => metrics.min_margin,
        "grad_norm" => grad_metrics.grad_norm,
        "response_norm" => grad_metrics.response_norm,
        "param_delta" => sqrt(sum(abs2, trainer.params.w .- initial_params.w) + sum(abs2, trainer.params.b .- initial_params.b)),
        "score_00" => metrics.means[1],
        "score_10" => metrics.means[2],
        "score_01" => metrics.means[3],
        "score_11" => metrics.means[4],
        "std_00" => metrics.stds[1],
        "std_10" => metrics.stds[2],
        "std_01" => metrics.stds[3],
        "std_11" => metrics.stds[4],
    )
end

function write_csv(path, rows)
    mkpath(dirname(path))
    isempty(rows) && return path
    keys_order = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(keys_order, ","))
        for row in rows
            println(io, join((row[k] for k in keys_order), ","))
        end
    end
    return path
end

function plot_rows(path, rows)
    mkpath(dirname(path))
    fig = Figure(size = (1400, 900))
    configs = unique(row["config"] for row in rows)
    metrics = [("mse", "readout MSE"), ("accuracy", "accuracy"), ("min_margin", "|score| margin"), ("grad_norm", "gradient norm")]
    for (i, (key, label)) in enumerate(metrics)
        ax = Axis(fig[cld(i, 2), mod1(i, 2)], title = label, xlabel = "epoch", ylabel = label)
        for cfg in configs
            sub = [row for row in rows if row["config"] == cfg]
            lines!(ax, [row["epoch"] for row in sub], [row[key] for row in sub], label = cfg)
        end
        axislegend(ax, position = :rt)
    end
    save(path, fig)
    return path
end

function write_parameter_svg(path, graph, config::LocalCheckerboardConfig)
    mkpath(dirname(path))
    weights = SparseArrays.getnzval(II.adj(graph))
    biases = II.getparam(graph.hamiltonian, II.MagField, :b)
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="780" height="260" viewBox="0 0 780 260">""")
        println(io, """<rect width="780" height="260" fill="white"/>""")
        println(io, """<text x="24" y="36" font-family="monospace" font-size="18">$(config.name)</text>""")
        entries = [
            "side=$(config.side), hidden=$(config.hidden_side), code=$(config.code_side), stride=$(config.code_stride)",
            "dynamics=$(config.dynamics_mode), T=$(config.temp), stepsize=$(config.stepsize), beta=$(config.β)",
            "doublewell barrier=$(config.doublewell_barrier), free T x$(config.free_temp_start_factor)->x$(config.free_temp_stop_factor), nudged T x$(config.nudged_temp_start_factor)->x$(config.nudged_temp_stop_factor)",
            "relaxation free/nudged=$(config.free_relaxation)/$(config.nudged_relaxation), Minit=$(config.minit)",
            "adjacency symmetry error=$(round(adjacency_symmetry_error(graph), sigdigits=4))",
            "weights: min=$(round(minimum(weights), digits=4)), max=$(round(maximum(weights), digits=4)), rms=$(round(sqrt(mean(abs2, weights)), digits=4))",
            "bias: min=$(round(minimum(biases), digits=4)), max=$(round(maximum(biases), digits=4)), rms=$(round(sqrt(mean(abs2, biases)), digits=4))",
        ]
        for (idx, line) in enumerate(entries)
            println(io, """<text x="24" y="$(70 + 30 * (idx - 1))" font-family="monospace" font-size="15">$line</text>""")
        end
        println(io, "</svg>")
    end
    return path
end

"""
    strip_weight_generators!(graph)

Remove construction-time random weight-generator closures before saving. The
trained adjacency is already stored in the graph; keeping anonymous generator
closures only makes JLD2 warn and is not needed to restore the trained model.
"""
function strip_weight_generators!(graph)
    for layerdata in getfield(graph, :layers)
        getfield(layerdata, :weightgenerator)[] = nothing
    end
    return graph
end

function print_metrics(epoch, metrics, grad)
    println(
        "epoch=$epoch mse=$(round(metrics.mse, digits=6)) acc=$(round(metrics.accuracy, digits=3)) ",
        "margin=$(round(metrics.min_margin, digits=6)) grad=$(round(grad.grad_norm, digits=6)) ",
        "scores=", round.(metrics.means, digits = 4),
    )
end

function run_config(config::LocalCheckerboardConfig, outdir)
    mkpath(outdir)
    graph = checkerboard_graph(config)
    layer = checkerboard_layer(graph, config)
    trainer = init_checker_trainer(layer, config; graph, optimiser = Optimisers.Adam(config.lr))
    x, y = xor_inputs_targets(config)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    initial_params = deepcopy(trainer.params)
    rows = Dict{String,Any}[]
    best_params = deepcopy(trainer.params)
    best_mse = Inf
    best_acc = -Inf
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))

    eval_seed = config.base_seed + 50_000_000
    metrics = evaluate_checker!(trainer, x, y, config; seed_offset = eval_seed)
    push!(rows, metric_row(config.name, 0, metrics, zero_grad, trainer, initial_params))
    print_metrics(0, metrics, zero_grad)
    if metrics.accuracy > best_acc || (metrics.accuracy == best_acc && metrics.mse < best_mse)
        best_acc, best_mse, best_params = metrics.accuracy, metrics.mse, deepcopy(trainer.params)
    end

    for epoch in 1:config.epochs
        grad = train_epoch!(trainer, x, y, batch_gradient, epoch, config)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_checker!(trainer, x, y, config; seed_offset = eval_seed)
            push!(rows, metric_row(config.name, epoch, metrics, grad, trainer, initial_params))
            print_metrics(epoch, metrics, grad)
            if metrics.accuracy > best_acc || (metrics.accuracy == best_acc && metrics.mse < best_mse)
                best_acc, best_mse, best_params = metrics.accuracy, metrics.mse, deepcopy(trainer.params)
            end
        end
    end

    trainer.params = best_params
    _broadcast_params!(trainer)
    strip_weight_generators!(trainer.prototype_graph)
    graph_path = II.save_isinggraph(joinpath(outdir, "$(config.name)_best_graph.jld2"), trainer.prototype_graph)
    svg_path = write_parameter_svg(joinpath(outdir, "$(config.name)_parameters.svg"), trainer.prototype_graph, config)
    close_checker_trainer!(trainer)
    return (; rows, graph_path, svg_path, best_mse, best_acc)
end

function default_configs()
    common = (;
        epochs = parse(Int, get(ENV, "ISING_LOCAL_XOR_EPOCHS", "1000")),
        log_every = parse(Int, get(ENV, "ISING_LOCAL_XOR_LOG_EVERY", "100")),
        minit = parse(Int, get(ENV, "ISING_LOCAL_XOR_MINIT", "4")),
        eval_repeats = parse(Int, get(ENV, "ISING_LOCAL_XOR_EVAL_REPEATS", "16")),
        workers = parse(Int, get(ENV, "ISING_LOCAL_XOR_THREADS", string(max(1, min(Threads.nthreads(), 8))))),
        free_relaxation = parse(Int, get(ENV, "ISING_LOCAL_XOR_FREE_RELAXATION", "100")),
        nudged_relaxation = parse(Int, get(ENV, "ISING_LOCAL_XOR_NUDGED_RELAXATION", "100")),
        β = parse(FT, get(ENV, "ISING_LOCAL_XOR_BETA", "0.05")),
        lr = parse(FT, get(ENV, "ISING_LOCAL_XOR_LR", "0.005")),
        weight_decay = parse(FT, get(ENV, "ISING_LOCAL_XOR_WEIGHT_DECAY", "1e-4")),
        temp = parse(FT, get(ENV, "ISING_LOCAL_XOR_TEMP", "0.005")),
        temp_is_factor = parse(Bool, get(ENV, "ISING_LOCAL_XOR_TEMP_IS_FACTOR", "false")),
        stepsize = parse(FT, get(ENV, "ISING_LOCAL_XOR_STEPSIZE", "0.05")),
        inter_weight_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_INTER_WEIGHT_SCALE", "0.05")),
        input_internal_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_INPUT_INTERNAL_SCALE", "0.02")),
        hidden_internal_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_HIDDEN_INTERNAL_SCALE", "0.02")),
        output_internal_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_OUTPUT_INTERNAL_SCALE", "0.02")),
        bias_scale = parse(FT, get(ENV, "ISING_LOCAL_XOR_BIAS_SCALE", "0.02")),
        weight_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_WEIGHT_SEED", "2")),
        internal_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_INTERNAL_SEED", "3")),
        bias_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_BIAS_SEED", "11")),
        base_seed = parse(Int, get(ENV, "ISING_LOCAL_XOR_BASE_SEED", "91000")),
        dynamics_mode = Symbol(get(ENV, "ISING_LOCAL_XOR_DYNAMICS", "langevin")),
        state_mode = Symbol(get(ENV, "ISING_LOCAL_XOR_STATE", "continuous")),
        init_mode = Symbol(get(ENV, "ISING_LOCAL_XOR_INIT", "random")),
        output_clamp_mode = Symbol(get(ENV, "ISING_LOCAL_XOR_OUTPUT_CLAMP", "readout")),
        doublewell_barrier = parse(FT, get(ENV, "ISING_LOCAL_XOR_DOUBLEWELL_BARRIER", "0.0")),
        free_temp_start_factor = parse(FT, get(ENV, "ISING_LOCAL_XOR_FREE_TEMP_START_FACTOR", "1.0")),
        free_temp_stop_factor = parse(FT, get(ENV, "ISING_LOCAL_XOR_FREE_TEMP_STOP_FACTOR", "1.0")),
        nudged_temp_start_factor = parse(FT, get(ENV, "ISING_LOCAL_XOR_NUDGED_TEMP_START_FACTOR", "1.0")),
        nudged_temp_stop_factor = parse(FT, get(ENV, "ISING_LOCAL_XOR_NUDGED_TEMP_STOP_FACTOR", "1.0")),
        temp_schedule_power = parse(FT, get(ENV, "ISING_LOCAL_XOR_TEMP_SCHEDULE_POWER", "1.0")),
    )
    return [
        LocalCheckerboardConfig(; name = "checker_2x2_global", side = 2, hidden_side = 2, code_side = 2, code_stride = 1, inter_radius = sqrt(2.0) + 1e-6, block_size = 4, common...),
        LocalCheckerboardConfig(; name = "checker_4x4_global", side = 4, hidden_side = 4, code_side = 4, code_stride = 1, inter_radius = sqrt(2.0) + 1e-6, block_size = 8, common...),
        LocalCheckerboardConfig(; name = "checker_8x8_global4", side = 8, hidden_side = 8, code_side = 8, code_stride = 1, inter_radius = sqrt(2.0) + 1e-6, block_size = 16, common...),
        LocalCheckerboardConfig(; name = "checker_8x8_inlaid4", side = 8, hidden_side = 8, code_side = 4, code_stride = 2, code_offset = (1, 1), inter_radius = sqrt(2.0) + 1e-6, block_size = 16, common...),
    ]
end

function selected_configs()
    wanted = split(get(ENV, "ISING_LOCAL_XOR_CONFIGS", "checker_2x2_global,checker_4x4_global"), ",")
    all = default_configs()
    return [cfg for cfg in all if cfg.name in wanted]
end

"""
    main()

Run the selected local checkerboard XOR experiments and write CSV, PNG, SVG and
JLD2 outputs under `ext/IsingLearning/experiments/local_checkerboard_xor/runs`.
Use `ISING_LOCAL_XOR_CONFIGS` to choose a comma-separated subset.
"""
function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_LOCAL_XOR_DIR", joinpath(DEFAULT_RUN_ROOT, "local_checkerboard_xor_$timestamp"))
    configs = selected_configs()
    isempty(configs) && error("No selected configs; check ISING_LOCAL_XOR_CONFIGS")
    all_rows = Dict{String,Any}[]
    summaries = String[]
    for cfg in configs
        println("Running $(cfg.name): ", cfg)
        result = run_config(cfg, joinpath(outdir, cfg.name))
        append!(all_rows, result.rows)
        push!(summaries, "- `$(cfg.name)`: best acc=$(round(result.best_acc, digits=3)), best mse=$(round(result.best_mse, digits=6)), graph=$(result.graph_path)")
    end
    csv_path = write_csv(joinpath(outdir, "local_checkerboard_xor_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "local_checkerboard_xor_progress.png"), all_rows)
    md_path = joinpath(outdir, "README.md")
    open(md_path, "w") do io
        println(io, "# Local Checkerboard XOR Run")
        println(io)
        println(io, "Inputs are two physical checkerboard freeze masks. No four-case one-hot input code is used.")
        println(io)
        println(io, "## Results")
        println(io, join(summaries, "\n"))
        println(io)
        println(io, "## Files")
        println(io, "- Metrics CSV: `$(basename(csv_path))`")
        println(io, "- Progress PNG: `$(basename(png_path))`")
        println(io, "- Per-config folders contain parameter SVGs and best graph JLD2 files.")
    end
    println("Saved metrics: $csv_path")
    println("Saved plot: $png_path")
    println("Saved run docs: $md_path")
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
