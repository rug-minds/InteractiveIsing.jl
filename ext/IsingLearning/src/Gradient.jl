"""
Contrastive gradient for IsingLayer from plus and minus nudged states
    Add them to the buffers, which are passed in as an optional argument for efficiency.

    This calculates ∂y/∂w, ∂y/∂b, and ∂y/∂α for the layer's learnable parameters.
    To get 
"""
function contrastive_gradient(layer::LayeredIsingGraphLayer, s_plus, s_minus, β::Real, buffers = nothing)
    if isnothing(buffers) # setup buffers
    end

    polynomial_ham = layer.hamiltonian[PolynomialHamiltonian]
    magfield = layer.hamiltonian[Magfield]
    bilinear = layer.hamiltonian[Bilinear]

    # Compute dH/dw!
    dw_thunk = @thunk begin
        parameter_derivative(bilinear, s_plus, dJ = buffers.w, buffermode = InteractiveIsing.OverwriteBuffer())
        parameter_derivative(bilinear, s_minus, dJ = buffers.w, buffermode = InteractiveIsing.SubtractBuffer())
        buffers.w ./= 2β
    end

    # Compute dH/db!
    parameter_derivative(magfield, s_plus, s_minus, db = buffers.b, buffermode = InteractiveIsing.OverwriteBuffer())
    parameter_derivative(magfield, s_minus, s_plus, db = buffers.b, buffermode = InteractiveIsing.SubtractBuffer())
    buffers.b ./= 2β
    # Compute dH/dα!
    parameter_derivative(polynomial_ham, s_plus, s_minus, dlp = buffers.α, buffermode = InteractiveIsing.OverwriteBuffer())
    parameter_derivative(polynomial_ham, s_minus, s_plus, dlp = buffers.α, buffermode = InteractiveIsing.SubtractBuffer())
    buffers.α ./= 2β

    (;w = buffers.w, b = buffers.b, α = buffers.α)    
end