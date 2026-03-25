## GAUSSIAN BERNOULLI
export GaussianBernoulli
"""
Gaussian Bernoulli:
H = sum_i (s_i^2*self_i/σ_i^2 + s_i*(sum_j w_ij*s_j + 2*μ_i*σ_i + b_i) + 1/2*s_i^2 + s_i*b_i)
"""
struct GaussianBernoulli <: HamiltonianTerm end
@inline reconstruct(hterm::GaussianBernoulli, g::AbstractIsingGraph) = hterm

@inline function calculate(::ΔH, hterm::GaussianBernoulli, hargs, proposal)
    s = hargs.s
    w = hargs.w
    self = hargs.self
    σ = hargs.σ
    μ = hargs.μ
    b = hargs.b

    j = at_idx(proposal)
    cum = zero(eltype(s))
    @turbo for ptr in nzrange(w, j)
        i = w.rowval[ptr]
        wij = w.nzval[ptr]
        cum += wij*s[i]
    end

    return (to_val(proposal)^2 - s[j]^2)*self[j]/σ[j]^2 + (s[j] - to_val(proposal))*(cum + 2*μ[j]*σ[j] + b[j]) + 1/2*(to_val(proposal)^2 - s[j]^2) + (s[j] - to_val(proposal))*b[j]
end

# function d_iH(::GaussianBernoulli, hargs, s_idx)
@inline function calculate(::d_iH, hterm::GaussianBernoulli, hargs, s_idx)
    s = hargs.s
    w = hargs.w
    self = hargs.self
    σ = hargs.σ
    μ = hargs.μ
    b = hargs.b

    cum = zero(eltype(s))
    @turbo for ptr in nzrange(w, s_idx)
        i = w.rowval[ptr]
        wij = w.nzval[ptr]
        cum += wij*s[i]
    end

    return 2*s[s_idx]*self[s_idx]/σ[s_idx]^2 + (cum + 2*μ[s_idx]*σ[s_idx] + b[s_idx]) + s[s_idx] + b[s_idx]
end
