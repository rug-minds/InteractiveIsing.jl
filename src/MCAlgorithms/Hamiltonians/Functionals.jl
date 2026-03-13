abstract type AbstractLinearFunctional end

struct ΔH <: AbstractLinearFunctional end
struct dH <: AbstractLinearFunctional end
struct H_i <: AbstractLinearFunctional end

## TODO: Whatever state is, needs to implement eltype
@inline function calculate(hF::AbstractLinearFunctional, hts::HTS, state::S, args...) where {HTS <: AbstractHamiltonianTerms, S <: AbstractIsingGraph}
    total = zero(eltype(state))
    total = @inline unrollreplace(total, hts...) do ftotal, hamiltonian
        ftotal = ftotal + @inline calculate(hF, hamiltonian, state, args...)
    end
    return total    
end
