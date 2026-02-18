abstract type HamiltonianCalulation end
struct ΔH <: HamiltonianCalulation end
# function ΔH(hterm::Hamiltonian, hargs, proposal)
#     error("No ΔH method defined for $(typeof(hterm)). Consider defining one or using parameterrefs to automatically generate ΔH.")
# end

const ΔH_expr = Dict{Type, Expr}()
get_ΔH_expr(::Type{H}) where {H} = ΔH_expr[Base.typename(H).wrapper]
get_ΔH_expr(h::H) where {H} = get_ΔH_expr(H)
gen_ΔH_expr = nothing
generated_func_calls = nothing

"""
Opt in flag to use paramref system
"""
# auto_generate(::ΔH, a) = false
"""
For auto generated ΔH, this function returns the parameterrefs to be used in the generated ΔH expression
"""
@inline ΔH_paramrefs(::Any) = nothing

@inline function calculate(::ΔH, hterm::HamiltonianTerm, hargs, proposal)
    return ΔH(hterm, hargs, proposal)
end

Base.@propagate_inbounds function calculate(::ΔH, HTS::AbstractHamiltonianTerms, hargs, proposal)
    ΔH_total = zero(eltype(hargs.s))
    ΔH_total = @inline unrollreplace(ΔH_total, HTS...) do Htot, hamiltonian
        Htot = Htot + @inline calculate(ΔH(), hamiltonian, hargs, proposal)
    end
    return ΔH_total
end

"""
From a function ΔH(::HamilTonianTermType, hargs, proposal)
    with as body an expression with indices (e.g. s[i]*w[i,j]*s[j])
    It will create an expression with parameterrefs that substitutes in the proposal (e.g. -s[i]*w[i,j]*s[j] -> (s[i]*w[i,j])*(s[j]-s'[j])
"""
macro Auto_ΔH(exp)
    @capture(exp, function Name_(::typename_, hargs_, proposal_) body_ end)
    @capture(body, :(return index_exp_))
    # println("Captured function: ", Name, " with typename: ", typename, " hargs: ", hargs, " proposal: ", proposal, " and body: ", body)
    # println("Captured index expression: ", index_exp)
    index_exp = nothing
    MacroTools.postwalk(body) do x
        if @capture(x, return rhs_)
            index_exp = rhs
        end
        x
    end
    !isnothing(index_exp) || error("The expression must contain an index expression of the form `index_exp_`.")

    q = quote
        @generated function $Name(::$typename, $hargs::H, $proposal) where H
            exp = to_delta_exp($(index_exp), $proposal)
            proposalname = $(QuoteNode(proposal))
            return quote
                hargs = (;hargs..., delta_1 = $proposalname)
                (@ParameterRefs $exp)(hargs; j = getidx($proposalname))
            end
        end
    end
    # println(q)
    esc(q)
end


## TODO: Check if multiple deltas actually work
# # """
# # Takes a hamiltonian H, a FlipProposal, idx_symbol: idx_value -> statevalue
# #     to generate the change in energy due to the FlipProposal
# # For Hamitlonians which have an explicit ΔH defined, it will get that function,
# # otherwise it will generate the ΔH based on the hamiltonian expression using parameterrefs
# # """
# export ΔH_exp
# function ΔH_exp(H::AbstractHamiltonianTerms, hargs, FlipProposals::FlipProposal...)
#     # has_methods = tuple() # Check which hamiltonians have explicit definitions
#     # generate_methods = tuple()
#     HS = H_types(H)

#     defined_ΔH_accesses = Expr[] # These hold the indices of hamiltonians with defined ΔH methods
#     delta_exps = Expr[] # These holds all the automatically generated delta expressions 
#                         # (e.g. delta rules applied to paramref expressions)
#     for (hidx, h) in enumerate(HS)
#         if hasmethod(ΔH, (h, Any, Any))
#             # has_methods = (has_methods..., h)
#             # push!(has_method_idxs, hidx)
#             push!(defined_ΔH_accesses, Expr(:ref, :H, hidx))
#         else
            
#             # generate_methods = (generate_methods..., h)

            
#             # Generate the deltas
#             push!(delta_exps, to_delta_exp(get_ΔH_expr(h), FlipProposals...))
            
#         end
#     end

#     # Defined expressions
#     pref_exprs = to_ParameterRefs.(delta_exps)

#     # For the generated methods
#     fixed_idxs, d_idxs = FlipProposals_to_fixed_idxs(FlipProposals...)

#     delta_names = ntuple(i -> Symbol("delta_", i), length(FlipProposals))

#     get_delta_index_exprs = map(i -> Expr(:ref, :(getidx($i))), delta_names)
#     global generated_func_calls = Expr(   :call, 
#                                     Expr(:call, +, pref_exprs...), 
#                                     Expr(:parameters, Expr.(Ref(:kw), fixed_idxs, get_delta_index_exprs)...), 
#                                     :hargs)

            

#     # For the defined methods
#     defined_func_calls = wrap_in_call(defined_ΔH_accesses, :ΔH, :hargs, :(FlipProposals...))
    
#     global gen_ΔH_expr = quote
#         ($(delta_names...),) = FlipProposals
#         # names = tuple($(QuoteNode.(delta_names...)))
#         hargs = (;hargs..., $(delta_names...) ) # Sets up FlipProposals to be used with parameter refs
#         $(Expr(:call, :+, defined_func_calls..., generated_func_calls))
#     end

#     return gen_ΔH_expr
# end


