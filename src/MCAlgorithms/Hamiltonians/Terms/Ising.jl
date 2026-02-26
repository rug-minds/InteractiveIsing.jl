const Ising{PV} = HamiltonianTerms(Quadratic, Bilinear, MagField{PV})

function Ising(g::AbstractIsingGraph, type::Symbol = :inactive_b)
    if type == :homogeneous_b
        return HamiltonianTerms(Quadratic(g), Bilinear(g), HomogeneousMagField(g))
    end
    b_active = type == :active_b

    HamiltonianTerms(Quadratic(g), Bilinear(g), MagField(g, b_active))
end

export Ising