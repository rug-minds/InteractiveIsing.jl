## GAUSSIAN BERNOULLI
export GaussianBernoulli
"""
Gaussian Bernoulli:
H = sum_i (s_i^2*self_i/σ_i^2 + s_i*(sum_j w_ij*s_j + 2*μ_i*σ_i + b_i) + 1/2*s_i^2 + s_i*b_i)
"""
struct GaussianBernoulli{P} <: HamiltonianTerm
    parameters::P
end

_default_gaussian_bernoulli_w(g) = adj(g)
_default_gaussian_bernoulli_self(g) = hasproperty(adj(g), :diag) && !isnothing(adj(g).diag) ? adj(g).diag : diag(adj(g))

function GaussianBernoulli(; w = nothing, self = nothing, σ = nothing, sigma = nothing, μ = nothing, mu = nothing, b = nothing)
    isnothing(σ) || isnothing(sigma) ||
        throw(ArgumentError("Pass either `σ` or `sigma`, not both."))
    isnothing(μ) || isnothing(mu) ||
        throw(ArgumentError("Pass either `μ` or `mu`, not both."))

    σ = isnothing(σ) ? sigma : σ
    μ = isnothing(μ) ? mu : μ

    params = Parameters(
        parameter(;
            w,
            type = AbstractSparseMatrix,
            default = _default_gaussian_bernoulli_w,
            ensure = ensure_isinggraph_adjacency,
            info = "Gaussian-Bernoulli coupling matrix w_ij",
        ),
        parameter(;
            self,
            type = AbstractVector,
            default = _default_gaussian_bernoulli_self,
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local quadratic self-coupling self_i",
        ),
        parameter(;
            σ,
            type = AbstractVector,
            default = ConstFill(1),
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local Gaussian standard deviation σ_i",
        ),
        parameter(;
            μ,
            type = AbstractVector,
            default = ConstFill(0),
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local Gaussian mean μ_i",
        ),
        parameter(;
            b,
            type = AbstractVector,
            default = ConstFill(0),
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local Bernoulli bias b_i",
        ),
    )
    return GaussianBernoulli(params)
end

@inline GaussianBernoulli(w, self, σ, μ, b) = GaussianBernoulli(; w, self, σ, μ, b)

@inline GaussianBernoulli(g::AbstractIsingGraph; kwargs...) = instantiate(GaussianBernoulli(; kwargs...), g)

@inline function _gaussian_bernoulli_weighted_sum(w, s, j)
    cum = zero(eltype(s))
    rowval = SparseArrays.getrowval(w)
    nzval = SparseArrays.getnzval(w)
    @turbo for ptr in nzrange(w, j)
        i = rowval[ptr]
        wij = nzval[ptr]
        cum += wij*s[i]
    end
    return cum
end

@inline function calculate(::ΔH, hterm::GaussianBernoulli, hargs, proposal)
    s = @inline graphstate(hargs)
    self = hterm.self
    σ = hterm.σ
    μ = hterm.μ
    b = hterm.b

    j = at_idx(proposal)
    cum = @inline _gaussian_bernoulli_weighted_sum(hterm.w, s, j)

    oldstate = s[j]
    newstate = to_val(proposal)
    Δs2 = newstate^2 - oldstate^2
    return Δs2*self[j]/σ[j]^2 + (oldstate - newstate)*(cum + 2*μ[j]*σ[j] + b[j]) + oftype(Δs2, 0.5)*Δs2 + (oldstate - newstate)*b[j]
end

# function d_iH(::GaussianBernoulli, hargs, s_idx)
@inline function calculate(::d_iH, hterm::GaussianBernoulli, hargs, s_idx)
    s = @inline graphstate(hargs)
    self = hterm.self
    σ = hterm.σ
    μ = hterm.μ
    b = hterm.b

    cum = @inline _gaussian_bernoulli_weighted_sum(hterm.w, s, s_idx)

    return 2*s[s_idx]*self[s_idx]/σ[s_idx]^2 + (cum + 2*μ[s_idx]*σ[s_idx] + b[s_idx]) + s[s_idx] + b[s_idx]
end

@inline function calculate(::H_i, hterm::GaussianBernoulli, model::S, idx) where {S <: AbstractIsingGraph}
    s = @inline graphstate(model)
    self = hterm.self
    σ = hterm.σ
    μ = hterm.μ
    b = hterm.b
    si = s[idx]
    cum = @inline _gaussian_bernoulli_weighted_sum(hterm.w, s, idx)

    return si^2*self[idx]/σ[idx]^2 + si*(cum + 2*μ[idx]*σ[idx] + b[idx]) + oftype(si, 0.5)*si^2 + si*b[idx]
end

@inline function _gaussian_bernoulli_w_buffer(w)
    rows = Int[]
    cols = Int[]
    vals = eltype(w)[]
    rowval = SparseArrays.getrowval(w)

    for col in axes(w, 2)
        for ptr in nzrange(w, col)
            push!(rows, rowval[ptr])
            push!(cols, col)
            push!(vals, zero(eltype(w)))
        end
    end

    return sparse(rows, cols, vals, size(w, 1), size(w, 2))
end

@inline function _gaussian_bernoulli_set_derivative!(::OverwriteBuffer, buffer, idx, value)
    buffer[idx] = value
end

@inline function _gaussian_bernoulli_set_derivative!(buffermode::AccumulateBuffer, buffer, idx, value)
    buffer[idx] += sign(buffermode) * value
end

@inline function _gaussian_bernoulli_set_derivative!(::OverwriteBuffer, buffer, i, j, value)
    buffer[i, j] = value
end

@inline function _gaussian_bernoulli_set_derivative!(buffermode::AccumulateBuffer, buffer, i, j, value)
    buffer[i, j] += sign(buffermode) * value
end

@inline function parameter_derivative(
    hterm::GaussianBernoulli,
    model::S;
    kwargs...
) where {S <: AbstractIsingGraph}
    return parameter_derivative(hterm, graphstate(model); kwargs...)
end

@inline function parameter_derivative(
    hterm::GaussianBernoulli,
    state::S;
    dw = _gaussian_bernoulli_w_buffer(hterm.w),
    dself = similar(hterm.self),
    dσ = similar(hterm.σ),
    dμ = similar(hterm.μ),
    db = similar(hterm.b),
    buffermode::BufferMode = OverwriteBuffer(),
) where {S <: AbstractVector}
    s = @inline state
    w = hterm.w
    self = hterm.self
    σ = hterm.σ
    μ = hterm.μ

    if buffermode isa OverwriteBuffer
        fill!(dw, zero(eltype(dw)))
    end

    rowval = SparseArrays.getrowval(w)
    @inbounds for col in axes(w, 2)
        sj = s[col]
        for ptr in nzrange(w, col)
            row = rowval[ptr]
            _gaussian_bernoulli_set_derivative!(buffermode, dw, row, col, s[row]*sj)
        end
    end

    @inbounds for i in eachindex(s)
        si = s[i]
        σi = σ[i]
        _gaussian_bernoulli_set_derivative!(buffermode, dself, i, si^2/σi^2)
        _gaussian_bernoulli_set_derivative!(buffermode, dσ, i, -2*si^2*self[i]/σi^3 + 2*μ[i]*si)
        _gaussian_bernoulli_set_derivative!(buffermode, dμ, i, 2*σi*si)
        _gaussian_bernoulli_set_derivative!(buffermode, db, i, 2*si)
    end

    return (; dw, dself, dσ, dμ, db)
end
