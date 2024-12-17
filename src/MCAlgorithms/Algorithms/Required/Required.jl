include("Utils.jl")
include("DeltaH.jl")
include("DerivativeH.jl")

_prepare(::Type{ConcreteHamiltonian}, ::Type{HType}) where HType <: Hamiltonian = _prepare(HType)
_prepare(::Type{Hamiltonian})= nothing