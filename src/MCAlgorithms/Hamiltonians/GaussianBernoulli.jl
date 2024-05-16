## GAUSSIAN BERNOULLI
export GaussianBernoulli
struct GaussianBernoulli <: Hamiltonian end

params(::Type{GaussianBernoulli}, GraphType::Type) = GatherHamiltonianParams(
                                (:σ, Vector{GraphType}, GraphType(1), "Standard Deviation"), 
                                (:μ, Vector{GraphType}, GraphType(0), "Mean"), 
                                (:b, Vector{GraphType}, GraphType(0), "Bias")
                                )

function Δi_H(::Type{GaussianBernoulli})
    collect_expr = :(w_ij*s_j)
    return_expr = :((sn_i^2-s_i^2)*self_i/σ_i^2+(s_i-sn_i)*(collect_expr+2*μ_i*σ_i)+1/2*(sn_i^2-s_i^2)+(s_i-sn_i)*b_i)
    return HExpression(collect_expr, return_expr)
end
