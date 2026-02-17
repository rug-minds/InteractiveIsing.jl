"""
H = Σ_ij J_ij s_i s_j

The Quadratic part of the Ising Hamiltonian
"""
struct Quadratic <: HamiltonianTerm end

Quadratic(type, len) = Quadratic()

# ΔH_paramrefs(::Quadratic) = @ParameterRefs (s[i]*w[i,j]*s[j] + self[j]*s[j]^2)


# @inline @Auto_ΔH function ΔH(::Quadratic, hargs, proposal)
#     return :(s[i]*w[i,j]*-s[j] + self[j]*s[j]^2)
# end

# const quadratic_exp = Ref(Expr(:block))
# @generated function ΔH(::Quadratic, hargs, proposal)
#     exp = to_delta_exp(:(s[i]*w[i,j]*-s[j] + self[j]*s[j]^2), proposal)
#     proposalname = :proposal
#     global quadratic_exp[] = quote
#             hargs = (; hargs..., delta_1 = $proposalname)
#             @ParameterRefs($exp)(hargs; j = getidx($proposalname))
#     end
#     return quadratic_exp[]
# end 

function ΔH(::Quadratic, hargs, proposal)
    s = hargs.s
    wij = hargs.w
    self = hargs.self
    j = at_idx(proposal)
    cum = zero(eltype(s))
    @turbo for ptr in nzrange(wij, j)
        i = wij.rowval[ptr]
        w = wij.nzval[ptr]
        cum += w*s[i]
    end

    ising_energy = cum*(s[j] - to_val(proposal)) # s - to_val because of the - sign

    return ising_energy + self[j]*(to_val(proposal)^2 - s[j]^2)

end
