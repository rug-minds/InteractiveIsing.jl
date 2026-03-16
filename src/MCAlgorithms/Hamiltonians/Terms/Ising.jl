const Ising{PV} = HamiltonianTerms(Quadratic, Bilinear, MagField{PV})

strip_nothing_kwargs(; kwargs...) =
    (; (k => v for (k, v) in pairs(kwargs) if v !== nothing)...)

@inline function Ising(;c = nothing, b = nothing, localpotential = nothing)
    quad_kwargs = strip_nothing_kwargs(;c)
    mag_kwargs = strip_nothing_kwargs(;b)
    lp_kwargs = strip_nothing_kwargs(;localpotential)
    return HamiltonianTerms(Quadratic(; quad_kwargs...), Bilinear(; lp_kwargs...), MagField(; mag_kwargs...))
    # if b == :homogeneous
    #     return HamiltonianTerms(Quadratic(; c = c), Bilinear(), MagField(active = true, homogeneous = true))
    # end
    # b_active = b == :active

    # return HamiltonianTerms(Quadratic(; c = c), Bilinear(), MagField(active = b_active))
end

@inline function Ising(g::AbstractIsingGraph; c = nothing, b = nothing, localpotential = nothing)
    return reconstruct(Ising(; c, b, localpotential), g)
end

@inline function reconstruct(hts::HamiltonianTerms, g::AbstractIsingGraph)
    return HamiltonianTerms((reconstruct.(hamiltonians(hts), Ref(g)))...)
end

export Ising
