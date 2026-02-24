const Ising{PV} = HamiltonianTerms(Quadratic, MagField{PV})

function Ising(g::AbstractIsingGraph, type::Symbol = :inactive_b)
    if type == :homogeneous_b
        return HamiltonianTerms(Quadratic(), HomogeneousMagField(g))
    end
    b_active = type == :active_b

    HamiltonianTerms(Quadratic(), MagField(g, b_active))
end

export Ising