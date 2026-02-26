abstract type AbstractLinearFunctional end

struct ΔH <: AbstractLinearFunctional end
struct dH <: AbstractLinearFunctional end

## TODO: Whatever state is, needs to implement eltype
@inline function calculate(hF::AbstractLinearFunctional, hts::HTS, state, args...) where {HTS <: AbstractHamiltonianTerms}
    total = zero(eltype(state))
    total = @inline unrollreplace(total, hts...) do ftotal, hamiltonian
        ftotal = ftotal + @inline calculate(hF, hamiltonian, state, args...)
    end
    return total    
end