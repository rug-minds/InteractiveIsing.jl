const Ising{PV} = HamiltonianTerms(Quadratic, MagField{PV})
Ising(g::AbstractIsingGraph, active = true) = Ising(eltype(g), statelen(g), active)
Ising(type, len, active = false) = HamiltonianTerms(Quadratic(), MagField(type, len, active))

export Ising