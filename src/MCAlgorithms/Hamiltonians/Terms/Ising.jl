const Ising{PV} = HamiltonianTerms(Quadratic, Bilinear, MagField{PV})

function Ising(g::AbstractIsingGraph; b = :inactive)
    if b == :homogeneous
        return HamiltonianTerms(Quadratic(g), Bilinear(g), HomogeneousMagField(g))
    end
    b_active = b == :active

    HamiltonianTerms(Quadratic(g), Bilinear(g), MagField(g, b_active))
end

export Ising