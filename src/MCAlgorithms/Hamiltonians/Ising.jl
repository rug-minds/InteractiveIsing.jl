const Ising{PV} = HamiltonianTerms(Linear, MagField{PV})
Ising(g) = Ising(eltype(g), statelen(g))
Ising(type, len, active = false) = HamiltonianTerms(Linear(), MagField(type, len, active))

export Ising