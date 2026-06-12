## GAUSSIAN-BERNOULLI RBM
export GaussianBernoulli,
    visible_indices,
    hidden_indices,
    joint_energy,
    marginal_energy,
    hidden_logits,
    hidden_probabilities,
    visible_energy_gradient,
    marginal_visible_gradient,
    sample_hidden!,
    sample_visible_given_hidden!

"""
    GaussianBernoulli(; w, mu, logσ2, b, visible_layer = 1, hidden_layer = 2)

Gaussian-Bernoulli restricted Boltzmann machine Hamiltonian from
Liao et al. (2022), with continuous visible units `v` and binary hidden
units `h`.

```math
E(v,h) =
    \\frac{1}{2}\\sum_i \\frac{(v_i - \\mu_i)^2}{\\sigma_i^2}
    - \\sum_{ij}\\frac{v_i}{\\sigma_i^2} W_{ij} h_j
    - \\sum_j b_j h_j.
```

The graph must have a continuous visible layer and a discrete hidden layer with
state set `{0, 1}`. `w` is visible-by-hidden. Variance is stored as
`logσ2`, matching the paper's non-negative variance parameterization.
"""
struct GaussianBernoulli{W,Mu,LogSigma2,B} <: HamiltonianTerm
    w::W
    μ::Mu
    logσ2::LogSigma2
    b::B
    visible_layer::Int
    hidden_layer::Int
end

function GaussianBernoulli(;
    w = nothing,
    W = nothing,
    μ = nothing,
    mu = nothing,
    logσ2 = nothing,
    logsigma2 = nothing,
    b = nothing,
    visible_layer::Integer = 1,
    hidden_layer::Integer = 2,
)
    isnothing(w) || isnothing(W) ||
        throw(ArgumentError("Pass either `w` or `W`, not both."))
    isnothing(μ) || isnothing(mu) ||
        throw(ArgumentError("Pass either `μ` or `mu`, not both."))
    isnothing(logσ2) || isnothing(logsigma2) ||
        throw(ArgumentError("Pass either `logσ2` or `logsigma2`, not both."))

    w = isnothing(w) ? W : w
    μ = isnothing(μ) ? mu : μ
    logσ2 = isnothing(logσ2) ? logsigma2 : logσ2

    return GaussianBernoulli(w, μ, logσ2, b, Int(visible_layer), Int(hidden_layer))
end

@inline GaussianBernoulli(w, μ, logσ2, b; visible_layer = 1, hidden_layer = 2) =
    GaussianBernoulli(; w, μ, logσ2, b, visible_layer, hidden_layer)

@inline GaussianBernoulli(g::AbstractIsingGraph; kwargs...) =
    instantiate(GaussianBernoulli(; kwargs...), g)

@inline visible_indices(hterm::GaussianBernoulli, model::AbstractIsingGraph) =
    graphidxs(model[hterm.visible_layer])

@inline hidden_indices(hterm::GaussianBernoulli, model::AbstractIsingGraph) =
    graphidxs(model[hterm.hidden_layer])

@inline function _gb_validate_layers(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    1 <= hterm.visible_layer <= length(model) ||
        throw(ArgumentError("visible_layer $(hterm.visible_layer) is outside the graph layer range."))
    1 <= hterm.hidden_layer <= length(model) ||
        throw(ArgumentError("hidden_layer $(hterm.hidden_layer) is outside the graph layer range."))
    hterm.visible_layer != hterm.hidden_layer ||
        throw(ArgumentError("visible_layer and hidden_layer must be different."))

    visible = model[hterm.visible_layer]
    hidden = model[hterm.hidden_layer]
    statetype(visible) isa Continuous ||
        throw(ArgumentError("GaussianBernoulli visible layer must be Continuous."))
    statetype(hidden) isa Discrete ||
        throw(ArgumentError("GaussianBernoulli hidden layer must be Discrete."))

    hstates = stateset(hidden)
    length(hstates) == 2 && iszero(first(hstates)) && last(hstates) == one(last(hstates)) ||
        throw(ArgumentError("GaussianBernoulli hidden layer state set must be {0, 1}; got $(hstates)."))
    return nothing
end

@inline function _gb_default_w(model::AbstractIsingGraph, visible_layer::Int, hidden_layer::Int)
    v_idxs = graphidxs(model[visible_layer])
    h_idxs = graphidxs(model[hidden_layer])
    return Matrix{eltype(model)}(adj(model)[v_idxs, h_idxs])
end

@inline _gb_resolve_value(x, model) = applicable(x, model) ? x(model) : x

function _gb_resolve_vector(x, model::AbstractIsingGraph, n::Integer, default_value, name::Symbol)
    T = eltype(model)
    x = isnothing(x) ? default_value : _gb_resolve_value(x, model)
    x = internalvalue(x, physicalunits(role = :dimensionless), physicalscales(model), model; parameter = name)
    if x isa Number
        return fill(convert(T, x), n)
    elseif x isa Base.RefValue
        return fill(convert(T, x[]), n)
    elseif x isa AbstractArray
        length(x) == n ||
            throw(DimensionMismatch("Expected GaussianBernoulli vector parameter of length $(n); got $(length(x))."))
        return collect(T, x)
    else
        throw(ArgumentError("Unsupported GaussianBernoulli vector parameter $(typeof(x))."))
    end
end

function _gb_resolve_matrix(x, model::AbstractIsingGraph, n_visible::Integer, n_hidden::Integer, visible_layer::Int, hidden_layer::Int)
    T = eltype(model)
    x = isnothing(x) ? _gb_default_w(model, visible_layer, hidden_layer) : _gb_resolve_value(x, model)
    x = internalvalue(x, physicalunits(role = :dimensionless), physicalscales(model), model; parameter = :w)
    x isa AbstractMatrix ||
        throw(ArgumentError("GaussianBernoulli `w` must resolve to an AbstractMatrix; got $(typeof(x))."))
    size(x) == (n_visible, n_hidden) ||
        throw(DimensionMismatch("GaussianBernoulli `w` must have size ($(n_visible), $(n_hidden)); got $(size(x))."))
    return Matrix{T}(x)
end

function instantiate(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    _gb_validate_layers(hterm, model)
    n_visible = length(visible_indices(hterm, model))
    n_hidden = length(hidden_indices(hterm, model))

    w = _gb_resolve_matrix(hterm.w, model, n_visible, n_hidden, hterm.visible_layer, hterm.hidden_layer)
    μ = _gb_resolve_vector(hterm.μ, model, n_visible, zero(eltype(model)), :μ)
    logσ2 = _gb_resolve_vector(hterm.logσ2, model, n_visible, zero(eltype(model)), :logσ2)
    b = _gb_resolve_vector(hterm.b, model, n_hidden, zero(eltype(model)), :b)

    return GaussianBernoulli(w, μ, logσ2, b, hterm.visible_layer, hterm.hidden_layer)
end

@inline _gb_visible_view(hterm::GaussianBernoulli, model::AbstractIsingGraph) =
    @view graphstate(model)[visible_indices(hterm, model)]

@inline _gb_hidden_view(hterm::GaussianBernoulli, model::AbstractIsingGraph) =
    @view graphstate(model)[hidden_indices(hterm, model)]

@inline _gb_sigmoid(x) = inv(one(x) + exp(-x))

@inline function _gb_softplus(x)
    T = typeof(x)
    return x > zero(T) ? x + log1p(exp(-x)) : log1p(exp(x))
end

function hidden_logits(hterm::GaussianBernoulli, v::AbstractVector)
    logits = similar(hterm.b)
    return hidden_logits!(logits, hterm, v)
end

function hidden_logits(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return hidden_logits(hterm, _gb_visible_view(hterm, model))
end

function hidden_logits!(logits::AbstractVector, hterm::GaussianBernoulli, v::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    length(v) == n_visible ||
        throw(DimensionMismatch("Visible vector length $(length(v)) does not match GaussianBernoulli visible size $(n_visible)."))
    length(logits) == n_hidden ||
        throw(DimensionMismatch("Logit buffer length $(length(logits)) does not match GaussianBernoulli hidden size $(n_hidden)."))

    @inbounds for j in 1:n_hidden
        total = hterm.b[j]
        for i in 1:n_visible
            total += hterm.w[i, j] * v[i] / exp(hterm.logσ2[i])
        end
        logits[j] = total
    end
    return logits
end

function hidden_probabilities(hterm::GaussianBernoulli, v::AbstractVector)
    probs = hidden_logits(hterm, v)
    @inbounds for i in eachindex(probs)
        probs[i] = _gb_sigmoid(probs[i])
    end
    return probs
end

function hidden_probabilities(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return hidden_probabilities(hterm, _gb_visible_view(hterm, model))
end

function joint_energy(hterm::GaussianBernoulli, v::AbstractVector, h::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    length(v) == n_visible ||
        throw(DimensionMismatch("Visible vector length $(length(v)) does not match GaussianBernoulli visible size $(n_visible)."))
    length(h) == n_hidden ||
        throw(DimensionMismatch("Hidden vector length $(length(h)) does not match GaussianBernoulli hidden size $(n_hidden)."))

    T = promote_type(eltype(v), eltype(h), eltype(hterm.w), eltype(hterm.μ), eltype(hterm.logσ2), eltype(hterm.b))
    total = zero(T)
    @inbounds for i in 1:n_visible
        σ2i = exp(hterm.logσ2[i])
        vi = v[i]
        total += T(0.5) * (vi - hterm.μ[i])^2 / σ2i
        wh = zero(T)
        for j in 1:n_hidden
            wh += hterm.w[i, j] * h[j]
        end
        total -= vi * wh / σ2i
    end
    @inbounds for j in 1:n_hidden
        total -= hterm.b[j] * h[j]
    end
    return total
end

function joint_energy(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return joint_energy(hterm, _gb_visible_view(hterm, model), _gb_hidden_view(hterm, model))
end

function marginal_energy(hterm::GaussianBernoulli, v::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    length(v) == n_visible ||
        throw(DimensionMismatch("Visible vector length $(length(v)) does not match GaussianBernoulli visible size $(n_visible)."))

    total = zero(promote_type(eltype(v), eltype(hterm.w), eltype(hterm.μ), eltype(hterm.logσ2), eltype(hterm.b)))
    @inbounds for i in 1:n_visible
        total += typeof(total)(0.5) * (v[i] - hterm.μ[i])^2 / exp(hterm.logσ2[i])
    end

    logits = hidden_logits(hterm, v)
    @inbounds for j in 1:n_hidden
        total -= _gb_softplus(logits[j])
    end
    return total
end

function marginal_energy(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return marginal_energy(hterm, _gb_visible_view(hterm, model))
end

function visible_energy_gradient!(grad::AbstractVector, hterm::GaussianBernoulli, v::AbstractVector, h::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    length(grad) == n_visible ||
        throw(DimensionMismatch("Gradient buffer length $(length(grad)) does not match GaussianBernoulli visible size $(n_visible)."))
    @inbounds for i in 1:n_visible
        wh = zero(eltype(grad))
        for j in 1:n_hidden
            wh += hterm.w[i, j] * h[j]
        end
        grad[i] = (v[i] - hterm.μ[i] - wh) / exp(hterm.logσ2[i])
    end
    return grad
end

function visible_energy_gradient(hterm::GaussianBernoulli, v::AbstractVector, h::AbstractVector)
    grad = similar(hterm.μ)
    return visible_energy_gradient!(grad, hterm, v, h)
end

function visible_energy_gradient(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return visible_energy_gradient(hterm, _gb_visible_view(hterm, model), _gb_hidden_view(hterm, model))
end

function marginal_visible_gradient!(grad::AbstractVector, hterm::GaussianBernoulli, v::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    length(grad) == n_visible ||
        throw(DimensionMismatch("Gradient buffer length $(length(grad)) does not match GaussianBernoulli visible size $(n_visible)."))
    probs = hidden_probabilities(hterm, v)
    @inbounds for i in 1:n_visible
        weighted_hidden = zero(eltype(grad))
        for j in 1:n_hidden
            weighted_hidden += hterm.w[i, j] * probs[j]
        end
        grad[i] = (v[i] - hterm.μ[i] - weighted_hidden) / exp(hterm.logσ2[i])
    end
    return grad
end

function marginal_visible_gradient(hterm::GaussianBernoulli, v::AbstractVector)
    grad = similar(hterm.μ)
    return marginal_visible_gradient!(grad, hterm, v)
end

function marginal_visible_gradient(hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return marginal_visible_gradient(hterm, _gb_visible_view(hterm, model))
end

function sample_hidden!(rng::AbstractRNG, h::AbstractVector, hterm::GaussianBernoulli, v::AbstractVector)
    probs = hidden_probabilities(hterm, v)
    @inbounds for j in eachindex(h)
        h[j] = rand(rng, eltype(h)) < probs[j] ? one(eltype(h)) : zero(eltype(h))
    end
    return h
end

function sample_hidden!(rng::AbstractRNG, hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return sample_hidden!(rng, _gb_hidden_view(hterm, model), hterm, _gb_visible_view(hterm, model))
end

function sample_visible_given_hidden!(rng::AbstractRNG, v::AbstractVector, hterm::GaussianBernoulli, h::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    @inbounds for i in 1:n_visible
        wh = zero(eltype(v))
        for j in 1:n_hidden
            wh += hterm.w[i, j] * h[j]
        end
        σ2i = exp(hterm.logσ2[i])
        v[i] = hterm.μ[i] + wh + sqrt(σ2i) * randn(rng, eltype(v))
    end
    return v
end

function sample_visible_given_hidden!(rng::AbstractRNG, hterm::GaussianBernoulli, model::AbstractIsingGraph)
    return sample_visible_given_hidden!(rng, _gb_visible_view(hterm, model), hterm, _gb_hidden_view(hterm, model))
end

@inline calculate(::H, hterm::GaussianBernoulli, model::S) where {S <: AbstractIsingGraph} =
    joint_energy(hterm, model)

@inline function _gb_local_visible_index(hterm::GaussianBernoulli, model::AbstractIsingGraph, idx::Integer)
    v_idxs = visible_indices(hterm, model)
    first(v_idxs) <= idx <= last(v_idxs) || return nothing
    return Int(idx - first(v_idxs) + 1)
end

@inline function _gb_local_hidden_index(hterm::GaussianBernoulli, model::AbstractIsingGraph, idx::Integer)
    h_idxs = hidden_indices(hterm, model)
    first(h_idxs) <= idx <= last(h_idxs) || return nothing
    return Int(idx - first(h_idxs) + 1)
end

function calculate(::ΔH, hterm::GaussianBernoulli, model::S, proposal) where {S <: AbstractIsingGraph}
    idx = at_idx(proposal)
    v = _gb_visible_view(hterm, model)
    h = _gb_hidden_view(hterm, model)
    old_state = from_val(proposal)
    new_state = to_val(proposal)

    visible_i = _gb_local_visible_index(hterm, model, idx)
    if !isnothing(visible_i)
        i = visible_i
        σ2i = exp(hterm.logσ2[i])
        wh = zero(eltype(model))
        @inbounds for j in eachindex(h)
            wh += hterm.w[i, j] * h[j]
        end
        return (new_state - hterm.μ[i])^2 / (2 * σ2i) -
               (old_state - hterm.μ[i])^2 / (2 * σ2i) -
               (new_state - old_state) * wh / σ2i
    end

    hidden_j = _gb_local_hidden_index(hterm, model, idx)
    if !isnothing(hidden_j)
        j = hidden_j
        logit = hterm.b[j]
        @inbounds for i in eachindex(v)
            logit += hterm.w[i, j] * v[i] / exp(hterm.logσ2[i])
        end
        return -(new_state - old_state) * logit
    end

    throw(ArgumentError("Proposal index $(idx) is not in the GaussianBernoulli visible or hidden layer."))
end

function calculate(::d_iH, hterm::GaussianBernoulli, model::S, proposal::SingleSpinProposal) where {S <: AbstractIsingGraph}
    s_idx = @inline at_idx(proposal)
    visible_i = _gb_local_visible_index(hterm, model, s_idx)
    isnothing(visible_i) &&
        throw(ArgumentError("GaussianBernoulli d_iH is only defined for visible continuous units; index $(s_idx) is not visible."))

    state = @inline graphstate(model)
    h = _gb_hidden_view(hterm, model)
    i = visible_i
    wh = zero(eltype(model))
    @inbounds for j in eachindex(h)
        wh += hterm.w[i, j] * h[j]
    end
    return ((@inline proposed_value(state, proposal)) - hterm.μ[i] - wh) / exp(hterm.logσ2[i])
end

function calculate(::H_i, hterm::GaussianBernoulli, model::S, idx) where {S <: AbstractIsingGraph}
    v = _gb_visible_view(hterm, model)
    h = _gb_hidden_view(hterm, model)

    visible_i = _gb_local_visible_index(hterm, model, idx)
    if !isnothing(visible_i)
        i = visible_i
        wh = zero(eltype(model))
        @inbounds for j in eachindex(h)
            wh += hterm.w[i, j] * h[j]
        end
        σ2i = exp(hterm.logσ2[i])
        return (v[i] - hterm.μ[i])^2 / (2 * σ2i) - v[i] * wh / σ2i
    end

    hidden_j = _gb_local_hidden_index(hterm, model, idx)
    if !isnothing(hidden_j)
        j = hidden_j
        return -hterm.b[j] * h[j]
    end

    throw(ArgumentError("Index $(idx) is not in the GaussianBernoulli visible or hidden layer."))
end

@inline function _gb_set_derivative!(::OverwriteBuffer, buffer, idx, value)
    buffer[idx] = value
end

@inline function _gb_set_derivative!(buffermode::AccumulateBuffer, buffer, idx, value)
    buffer[idx] += sign(buffermode) * value
end

@inline function _gb_set_derivative!(::OverwriteBuffer, buffer, i, j, value)
    buffer[i, j] = value
end

@inline function _gb_set_derivative!(buffermode::AccumulateBuffer, buffer, i, j, value)
    buffer[i, j] += sign(buffermode) * value
end

@inline function parameter_derivative(
    hterm::GaussianBernoulli,
    model::S;
    kwargs...
) where {S <: AbstractIsingGraph}
    return parameter_derivative(hterm, _gb_visible_view(hterm, model), _gb_hidden_view(hterm, model); kwargs...)
end

function parameter_derivative(
    hterm::GaussianBernoulli,
    v::AbstractVector,
    h::AbstractVector;
    dw = similar(hterm.w),
    dμ = similar(hterm.μ),
    dlogσ2 = similar(hterm.logσ2),
    db = similar(hterm.b),
    buffermode::BufferMode = OverwriteBuffer(),
)
    n_visible, n_hidden = size(hterm.w)
    @inbounds for i in 1:n_visible
        σ2i = exp(hterm.logσ2[i])
        wh = zero(eltype(dw))
        for j in 1:n_hidden
            _gb_set_derivative!(buffermode, dw, i, j, -v[i] * h[j] / σ2i)
            wh += hterm.w[i, j] * h[j]
        end
        _gb_set_derivative!(buffermode, dμ, i, (hterm.μ[i] - v[i]) / σ2i)
        _gb_set_derivative!(buffermode, dlogσ2, i, -((v[i] - hterm.μ[i])^2) / (2 * σ2i) + v[i] * wh / σ2i)
    end
    @inbounds for j in 1:n_hidden
        _gb_set_derivative!(buffermode, db, j, -h[j])
    end
    return (; dw, dμ, dlogσ2, db)
end
