"""
    accumulate_symmetric_contrastive_gradient!(graph, s_plus, s_minus, buffers)

Accumulate the unscaled symmetric contrastive `dH(s_plus) - dH(s_minus)`
gradient directly into `buffers`. This computes the product differences in one
pass over the sparse coupling topology instead of accumulating the plus and
minus Hamiltonian derivatives in two separate passes.
"""
function accumulate_symmetric_contrastive_gradient!(
    graph::G,
    s_plus::SP,
    s_minus::SM,
    buffers::B,
) where {G,SP<:AbstractVector,SM<:AbstractVector,B}
    polynomial_ham = hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)
    magfield = _mnist_base_magfield(graph)
    bilinear = graph.hamiltonian[InteractiveIsing.Bilinear]
    adjacency = bilinear.J
    rows = SparseArrays.rowvals(adjacency)
    half = eltype(buffers.w)(0.5)

    # Recurrent couplings: dJ += -0.5 * (p_i*p_j - m_i*m_j).
    @inbounds for col in 1:size(adjacency, 2)
        for ptr in SparseArrays.nzrange(adjacency, col)
            row = rows[ptr]
            buffers.w[ptr] += -half * (
                eltype(buffers.w)(s_plus[row]) * eltype(buffers.w)(s_plus[col]) -
                eltype(buffers.w)(s_minus[row]) * eltype(buffers.w)(s_minus[col])
            )
        end
    end

    # Biases: db += -c * (p_i - m_i).
    c = eltype(buffers.b)(magfield.c)
    @inbounds for idx in eachindex(buffers.b)
        buffers.b[idx] += -c * (eltype(buffers.b)(s_plus[idx]) - eltype(buffers.b)(s_minus[idx]))
    end

    # Local polynomial terms are uncommon for MNIST, but keep them direct too.
    if !isnothing(polynomial_ham)
        hasproperty(buffers, :α) || error("graph has Quadratic local potential but gradient buffers have no α")
        coeff = eltype(buffers.α)(polynomial_ham.c[])
        power = InteractiveIsing.order(polynomial_ham)
        @inbounds for idx in eachindex(buffers.α)
            buffers.α[idx] += coeff * (
                eltype(buffers.α)(s_plus[idx])^power -
                eltype(buffers.α)(s_minus[idx])^power
            )
        end
    end

    hasproperty(buffers, :α) && return (; w = buffers.w, b = buffers.b, α = buffers.α)
    return (; w = buffers.w, b = buffers.b)
end

"""
    contrastive_gradient_new(graph, s_plus, s_minus, β; buffers)

Opt-in symmetric contrastive gradient with the same public call shape as
`contrastive_gradient`, but with direct product-difference accumulation for the
plus-minus contrast. The `β` argument is accepted for API parity; callers keep
doing minibatch scaling outside this raw accumulator.
"""
function contrastive_gradient_new(graph, s_plus, s_minus, β::Real; buffers = nothing)
    if isnothing(buffers) # setup buffers
    end
    return accumulate_symmetric_contrastive_gradient!(graph, s_plus, s_minus, buffers)
end

"""
Contrastive gradient for IsingLayer from plus and minus nudged states
    Add them to the buffers, which are passed in as an optional argument for efficiency.

    This calculates ∂y/∂w, ∂y/∂b, and ∂y/∂α for the layer's learnable parameters.

    The buffers should be of the shape
        - getnzval(adj(g))
        - nstates(g)
        - nstates(g)
"""


function contrastive_gradient(graph, s_plus, s_minus, β::Real; buffers = nothing)
    if isnothing(buffers) # setup buffers
    end

    polynomial_ham = hamiltonian_or_nothing(graph.hamiltonian, InteractiveIsing.Quadratic)
    magfield = _mnist_base_magfield(graph)
    bilinear = graph.hamiltonian[InteractiveIsing.Bilinear]

    # Laborieux et al. 2021 use the symmetric EP estimator
    # (∂Φ(sβ)/∂θ - ∂Φ(s-β)/∂θ) / 2β. With their nudging convention this
    # estimates the *negative* loss gradient, i.e. the update direction for
    # θ += η * direction. Our dynamics minimize H + βC, so Φ = -H.
    # Therefore the loss gradient to pass to `Optimisers.update`, which applies
    # θ -= η * gradient, is (∂H(sβ)/∂θ - ∂H(s-β)/∂θ) / 2β.
    # Accumulate dH/dw (plus - minus) into buffers.
    InteractiveIsing.parameter_derivative(bilinear, s_plus, dJ = buffers.w, buffermode = InteractiveIsing.AccumulateBuffer{+}())
    InteractiveIsing.parameter_derivative(bilinear, s_minus, dJ = buffers.w, buffermode = InteractiveIsing.SubtractBuffer())

    # Accumulate dH/db
    InteractiveIsing.parameter_derivative(magfield, s_plus, db = buffers.b, buffermode = InteractiveIsing.AccumulateBuffer{+}())
    InteractiveIsing.parameter_derivative(magfield, s_minus, db = buffers.b, buffermode = InteractiveIsing.SubtractBuffer())

    # Accumulate dH/dα only for graphs that actually have a local-potential term.
    if !isnothing(polynomial_ham)
        hasproperty(buffers, :α) || error("graph has Quadratic local potential but gradient buffers have no α")
        InteractiveIsing.parameter_derivative(polynomial_ham, s_plus, dlp = buffers.α, buffermode = InteractiveIsing.AccumulateBuffer{+}())
        InteractiveIsing.parameter_derivative(polynomial_ham, s_minus, dlp = buffers.α, buffermode = InteractiveIsing.SubtractBuffer())
    end

    hasproperty(buffers, :α) && return (; w = buffers.w, b = buffers.b, α = buffers.α)
    return (; w = buffers.w, b = buffers.b)
end
