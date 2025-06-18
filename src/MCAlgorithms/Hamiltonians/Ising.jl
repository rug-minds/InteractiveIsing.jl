const Ising{PV} = HamiltonianTerms(Linear, MagField{PV})
Ising(g) = Ising(eltype(g), statelen(g))
Ising(type, len) = HamiltonianTerms(Linear(), MagField(type, len))

export Ising