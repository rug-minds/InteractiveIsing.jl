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
