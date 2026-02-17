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

const quadratic_exp = Ref(Expr(:block))
@generated function ΔH(::Quadratic, hargs, proposal)
    exp = to_delta_exp(:(s[i]*w[i,j]*-s[j] + self[j]*s[j]^2), proposal)
    proposalname = :proposal
    global quadratic_exp[] = quote
            hargs = (; hargs..., delta_1 = $proposalname)
            @ParameterRefs($exp)(hargs; j = getidx($proposalname))
    end
    return quadratic_exp[]
end 
