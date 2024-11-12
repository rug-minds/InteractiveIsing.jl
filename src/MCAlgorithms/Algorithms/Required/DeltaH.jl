# dtype := dispatchtype
struct Δi_H <: ConcreteHamiltonian end
Δi_H(::Type{H}) where H <: Hamiltonian  = Δi_H(H())

function reserved_symbols(::Δi_H)
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

args(::Δi_H) = (:i, :gstate, :newstate, :oldstate, :gadj, :gparams, :dtype)

# prepare(::Type{Δi_H}) = nothing

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
    # collect_1 = zero(eltype(g)); collect_2 = zero(eltype(g)); ...
    collects_init = ("collect_$i = zero(eltype(gstate)) $(i != num_collects ? ';' : '\n')" for i in 1:num_collects if i != no_collect_idx)

    # Create all terms to be collected in the for loop
    #   collect_1 += wij*expr1
    #   collect_2 += wij*expr2
    collects = ("collect_$idx += $(collect_exprs[unique_idxs[idx]])" for idx in eachindex(unique_idxs) if unique_idxs[idx] != no_collect_idx)

    # Substitute in exprs
    subs_return_exprs = map(
        (expr, i) -> symbwalk!(x -> x == :collect_expr ? Symbol("collect_",i) : x, expr),
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

    
    return body
end

function replace_macrosymb(expr, symb, replace...)
    exprc = deepcopy(expr)
    idxs = find_symb(expr, symb)
    println(idxs)
    idxs = idxs[1:end-1]
    
    splice!(enter_args(exprc, idxs[1:end-1]...).args, idxs[end]:idxs[end], [replace...])
    return exprc
end
export replace_macrosymb

function H_Macros(::Δi_H, symbol, hamiltonians::Hamiltonian)
    if symbol == Symbol("@initialize")
        exp = Expr(:block)
        for hidx in 1:length(hamiltonians)
            push!(exp.args, :(collect_$hidx = zero(eltype(gstate))))
        end
        return exp
    elseif symbol == Symbol("@collect")
        exp = Expr(:block)
        for h in hamiltonians
            c_exp = Δi_H(h).collect_expr
            push!(exp.args, c_exp)
        end
    elseif symbol == Symbol("@collectsum")
        exp = Expr(:block)
        for h in hamiltonians
            c_exp = Δi_H(h).return_expr
            push!(exp.args, c_exp)
        end
    end
end

function replace_H_macros!(::Δi_H, exp; hamiltonians)
    symbols = [Symbol("@initialize"), Symbol("@collect"), Symbol("@collectsum")]
    symbwalk!(x -> x in symbols ? H_Macros(Δi_H(), x, hamiltonians...) : x, exp)
end

macro ConcreteHamiltonian(expr)
    @capture(expr, function fname_(a__) body_ end)
    if isnothing(body)
        @capture(expr, function fname_(a__) where T_ body_ end)
        if isnothing(body)
            @capture(expr, function fname_(a__) where {T__} body_ end)
        end
    end
    paramname = find_type_in_args(a, :Parameters)
    c_exp = replace_reserved!(expr, (@eval $(fname)()))
    # Replace the symbols that are indexed by their vector form
    c_exp = replace_indices(c_exp)
    f_exp = GeneratedParametersExp(fname, c_exp, paramname, a, (exp, parmname; hamiltonians) -> replace_H_macros!(Δi_H(), expr; hamiltonians))
    f_gen = GeneratedParametersGen(fname, a)

    return esc(quote
        $f_exp
        $f_gen
    end)

end

# @ConcreteHamiltonian function Δi_H(i, gstate::Vector{T}, newstate, oldstate, gadj, gparams, layertype) where T
#             # Collect the initial energy
#             @initialize
#             @turbo check_empty = true for ptr in nzrange(gadj, i)
#                 j = gadj.rowval[ptr]
#                 wij = gadj.nzval[ptr]

#                 @collect
#             end
            
#             return @collectsum
# end


