include("DeltaH.jl")
include("DerivativeH.jl")

_prepare(::Type{DerivedHamiltonian}, ::Type{HType}) where HType <: Hamiltonian = _prepare(HType)
_prepare(::Type{Hamiltonian})= nothing