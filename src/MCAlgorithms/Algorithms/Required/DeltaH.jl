# dtype := dispatchtype
struct Δi_H <: DerivedHamiltonian end

args(::Type{Δi_H}) = (:i, :gstate, :newstate, :oldstate, :gadj, :gparams, :dtype)

# TODO: Remove export
export H_expr
function H_expr(::Type{Δi_H}, graph, hamiltonians::Type{<:Hamiltonian}...)
    hexprs = Δi_H.(hamiltonians)
    collect_exprs = getfield.(hexprs, :collect_expr)

    return_exprs = getfield.(hexprs, :return_expr)

    # Identify the unique collect expressions
    num_collects, element_identifier, unique_idxs = identify_unique_elements(collect_exprs)
    
    # Don't collect the empty expressions
    no_collect_idx = findfirst(x -> x == :(), collect_exprs)
    
    # Init all collect_i variables to zero
    # collect_1 = zero(eltype(g)); collect_2 = zero(eltype(gstate)); ...
    collects_init = ("collect_$i = zero(eltype(gstate)) $(i != num_collects ? ';' : '\n')" for i in 1:num_collects if i != no_collect_idx)

    # Create all terms to be collected in the for loop
    #   collect_1 += wij*expr1
    #   collect_2 += wij*expr2
    collects = ("collect_$idx += $(collect_exprs[unique_idxs[idx]])" for idx in eachindex(unique_idxs) if unique_idxs[idx] != no_collect_idx)

    # Substitute in exprs
    subs_return_exprs = map(
        (expr, i) -> symbwalk(x -> x == :collect_expr ? Symbol("collect_",i) : x, expr),
        return_exprs, element_identifier    
    )
    
    # Main body with SIMD collect from the graph
    body = "function H(i, gstate::Vector{T}, newstate, oldstate, gadj, gparams, layertype) where T
            # Collect the initial energy
            $(collects_init...)
            @turbo check_empty = $(check_empty[]) for ptr in nzrange(gadj, i)
                j = gadj.rowval[ptr]
                wij = gadj.nzval[ptr]

                $(collects...)
            end
            
            return $(join(string.(subs_return_exprs), " + "))
        end"

    body = replace_inactive_symbs(graph.params, Meta.parse(body))
    body = replace_reserved(Metropolis, body)
    return body
end