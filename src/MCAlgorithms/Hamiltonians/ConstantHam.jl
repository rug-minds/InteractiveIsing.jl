struct ConstantHam{T} <: Hamiltonian
    value::T
end

ΔH(ch::ConstantHam, hargs, delta) = ch.value