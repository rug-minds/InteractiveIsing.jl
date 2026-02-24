abstract type AbstractHamiltonianFunctional end

struct ΔH <: AbstractHamiltonianFunctional end
struct dH <: AbstractHamiltonianFunctional end

@inline function calculate(hF::AbstractHamiltonianFunctional, hts::HTS, hargs::HArgs, args...) where {HTS <: AbstractHamiltonianTerms, HArgs}
    total = zero(eltype(hargs.s))
    total = @inline unrollreplace(total, hts...) do ftotal, hamiltonian
        ftotal = ftotal + @inline calculate(hF, hamiltonian, hargs, args...)
    end
    return total    
end