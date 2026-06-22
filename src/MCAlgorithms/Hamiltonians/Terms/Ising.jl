const Ising{PV} = HamiltonianTerms(Quadratic, Bilinear, ExtField{PV})

strip_nothing_kwargs(; kwargs...) =
    (; (k => v for (k, v) in pairs(kwargs) if v !== nothing)...)

@inline function Ising(;c = nothing, b = nothing, adj = nothing, localpotential = nothing)
    quad_kwargs = strip_nothing_kwargs(;c, localpotential)
    field_kwargs = strip_nothing_kwargs(;b)
    adj_kwargs = strip_nothing_kwargs(;adj)
    return HamiltonianTerms(Quadratic(; quad_kwargs...), Bilinear(; adj_kwargs...), ExtField(; field_kwargs...))
    # if b == :homogeneous
    #     return HamiltonianTerms(Quadratic(; c = c), Bilinear(), ExtField(active = true, homogeneous = true))
    # end
    # b_active = b == :active

    # return HamiltonianTerms(Quadratic(; c = c), Bilinear(), ExtField(active = b_active))
end

@inline function Ising(g::AbstractIsingGraph; c = nothing, b = nothing, adj = nothing, localpotential = nothing)
    return instantiate(Ising(; c, b, adj, localpotential), g)
end

@inline function instantiate(hts::HamiltonianTerms, g)
    return HamiltonianTerms((instantiate.(hamiltonians(hts), Ref(g)))...)
end

export Ising
