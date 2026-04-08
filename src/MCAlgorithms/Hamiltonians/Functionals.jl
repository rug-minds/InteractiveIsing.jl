abstract type AbstractLinearFunctional end

struct H <: AbstractLinearFunctional end
struct ΔH <: AbstractLinearFunctional end


"""
Energy derivative w.r.t a single unit, i.e. ∂H/∂s_i
"""
struct d_iH <: AbstractLinearFunctional end

"""
The energy of a single unit
"""
struct H_i <: AbstractLinearFunctional end 

## TODO: Whatever state is, needs to implement eltype
@inline function calculate(hF::AbstractLinearFunctional, hts::HTS, state::S, args...) where {HTS <: AbstractHamiltonianTerms, S <: AbstractIsingGraph}
    total = zero(eltype(state))
    total = @inline unrollreplace(total, hts...) do ftotal, hamiltonian
        ftotal = ftotal + @inline calculate(hF, hamiltonian, state, args...)
    end
    return total    
end
