
const ΔH_expr = Dict{Type, Expr}()
get_ΔH_expr(::Type{H}) where {H} = ΔH_expr[Base.typename(H).wrapper]
get_ΔH_expr(h::H) where {H} = get_ΔH_expr(H)
gen_ΔH_expr = nothing
generated_func_calls = nothing

"""
Takes a hamiltonian H, a FlipProposal, idx_symbol: idx_value -> statevalue
    to generate the change in energy due to the FlipProposal
For Hamitlonians which have an explicit ΔH defined, it will get that function,
otherwise it will generate the ΔH based on the hamiltonian expression using parameterrefs
"""
@generated function ΔH(H::AbstractHamiltonianTerms, hargs, FlipProposals::FlipProposal...)
    # has_methods = tuple() # Check which hamiltonians have explicit definitions
    # generate_methods = tuple()
    HS = H_types(H)

    defined_ΔH_accesses = Expr[] # These hold the indices of hamiltonians with defined ΔH methods
    delta_exps = Expr[] # These holds all the automatically generated delta expressions 
                        # (e.g. delta rules applied to paramref expressions)
    for (hidx, h) in enumerate(HS)
        if hasmethod(ΔH, (h, Any, Any))
            # has_methods = (has_methods..., h)
            # push!(has_method_idxs, hidx)
            push!(defined_ΔH_accesses, Expr(:ref, :H, hidx))
        else
            
            # generate_methods = (generate_methods..., h)

            
            # Generate the deltas
            push!(delta_exps, to_delta_exp(get_ΔH_expr(h), FlipProposals...))
            
        end
    end

    # Defined expressions
    pref_exprs = to_ParameterRefs.(delta_exps)

    # For the generated methods
    fixed_idxs, d_idxs = FlipProposals_to_fixed_idxs(FlipProposals...)

    delta_names = ntuple(i -> Symbol("delta_", i), length(FlipProposals))

    get_delta_index_exprs = map(i -> Expr(:ref, :(getidx($i))), delta_names)
    global generated_func_calls = Expr(   :call, 
                                    Expr(:call, +, pref_exprs...), 
                                    Expr(:parameters, Expr.(Ref(:kw), fixed_idxs, get_delta_index_exprs)...), 
                                    :hargs)

            

    # For the defined methods
    defined_func_calls = wrap_in_call(defined_ΔH_accesses, :ΔH, :hargs, :(FlipProposals...))
    
    global gen_ΔH_expr = quote
        ($(delta_names...),) = FlipProposals
        # names = tuple($(QuoteNode.(delta_names...)))
        hargs = (;hargs..., $(delta_names...) ) # Sets up FlipProposals to be used with parameter refs
        $(Expr(:call, :+, defined_func_calls..., generated_func_calls))
    end

    return gen_ΔH_expr
end



