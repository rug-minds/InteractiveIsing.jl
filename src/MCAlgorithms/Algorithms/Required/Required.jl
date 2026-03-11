include("Utils.jl")
include("DeltaH.jl")
include("DerivativeH.jl")

@inline _prepare(::Type{ConcreteHamiltonian}, ::Type{HType}) where HType <: Hamiltonian = _prepare(HType)
@inline _prepare(::Type{Hamiltonian}) = nothing
