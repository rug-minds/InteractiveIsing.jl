const Ising{PV} = HamiltonianTerms(Quadratic, Bilinear, MagField{PV})

@inline function Ising(; b = :inactive)
    if b == :homogeneous
        return HamiltonianTerms(Quadratic(), Bilinear(), MagField(active = true, homogeneous = true))
    end
    b_active = b == :active

    return HamiltonianTerms(Quadratic(), Bilinear(), MagField(active = b_active))
end

@inline function Ising(g::AbstractIsingGraph; b = :inactive)
    return reconstruct(Ising(; b), g)
end

@inline function reconstruct(hts::HamiltonianTerms, g::AbstractIsingGraph)
    return HamiltonianTerms((reconstruct.(hamiltonians(hts), Ref(g)))...)
end

export Ising
