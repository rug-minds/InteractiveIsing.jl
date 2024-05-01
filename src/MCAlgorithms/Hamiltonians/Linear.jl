struct Linear <: Hamiltonian end
params(::Type{Linear}) = nothing

function Δi_H(::Type{Linear})
    return 
end