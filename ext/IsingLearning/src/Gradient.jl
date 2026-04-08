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

    polynomial_ham = graph.hamiltonian[InteractiveIsing.PolynomialHamiltonian]
    magfield = graph.hamiltonian[InteractiveIsing.MagField]
    bilinear = graph.hamiltonian[InteractiveIsing.Bilinear]

    # Compute dH/dw!
    # dw_thunk = @thunk begin
        InteractiveIsing.parameter_derivative(bilinear, s_plus, dJ = buffers.w, buffermode = InteractiveIsing.OverwriteBuffer())
        InteractiveIsing.parameter_derivative(bilinear, s_minus, dJ = buffers.w, buffermode = InteractiveIsing.SubtractBuffer())
        buffers.w ./= 2β
    # end

    # Compute dH/db!
    InteractiveIsing.parameter_derivative(magfield, s_plus, db = buffers.b, buffermode = InteractiveIsing.OverwriteBuffer())
    InteractiveIsing.parameter_derivative(magfield, s_minus, db = buffers.b, buffermode = InteractiveIsing.SubtractBuffer())
    buffers.b ./= 2β
    # Compute dH/dα!
    InteractiveIsing.parameter_derivative(polynomial_ham, s_plus, dlp = buffers.α, buffermode = InteractiveIsing.OverwriteBuffer())
    InteractiveIsing.parameter_derivative(polynomial_ham, s_minus, dlp = buffers.α, buffermode = InteractiveIsing.SubtractBuffer())
    buffers.α ./= 2β

    (;w = buffers.w, b = buffers.b, α = buffers.α)    
end