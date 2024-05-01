
"""
Creates a bracketed string of the arguments from the args function for a hamiltonian
"""
H_args(::Type{H}) where H = join(args(H), ", ")

"""
splice! for strings
"""
function strsplice(original_string::AbstractString, start_index::Int, end_index::Int, replacement_substring::AbstractString)
    # Extract substrings before and after the indices
    substring_before = original_string[1:start_index - 1]
    substring_after = original_string[end_index + 1:end]

    # Construct the modified string
    new_string = substring_before * replacement_substring * substring_after
    return new_string
end

"""
In a symbol, find the pattern _(combination of symbols)
"""
function find_i_symb(symb)
    if !isa(symb, Symbol)
        return nothing, nothing
    end
    str = String(symb)
    symbs = Char[]
    found_underscore = findfirst(x -> x == '_', str)
    found_last_symb = found_underscore
    if !isnothing(found_underscore)
        while (found_last_symb+1 <= length(str)) && (str[found_last_symb+1] in ['i','j','k','l','m','n'])
            found_last_symb +=1
            push!(symbs, str[found_last_symb])
        end

        # If found any symbols
        if length(symbs) > 0
            return found_underscore:found_last_symb, symbs
        end
    end

    return nothing, nothing

end

"""
Replace the _(symbs) with [symb1,symb2,...]
"""
function i_to_vec_expr(symb)
    idxs, symbs = find_i_symb(symb)
    if isnothing(idxs)
        return nothing
    end
    vecstr = "[$(join(symbs, ","))]"
    newstr = String(symb)
    str = strsplice(newstr, first(idxs), last(idxs), vecstr)
    return Meta.parse(str)
end



"""
Walk through all symbols in an expression and replace them by the function func
"""
symbwalk(func, expr::Expr) = _symbwalk(func, deepcopy(expr))
function _symbwalk(func, expr::Expr)
    for arg_idx in eachindex(expr.args)
        if expr.args[arg_idx] isa Expr
            _symbwalk(func, expr.args[arg_idx])
        else
            expr.args[arg_idx] = func(expr.args[arg_idx])
        end
    end
    return expr
end

"""
From a vector of the collect expressions for the hamiltonians
    find out the unique expressions and the indices of the unique expressions
"""
function identify_unique_elements(els)
    element_identifier = Int[1:length(els);]

    unique_els = length(els)
    unique_idxs = Int[]
    for i in eachindex(els)
        # If element was already identified
        #
        if element_identifier[i] != i
            continue
        end

        for j in i+1:length(els)
            if els[i] == els[j]
                    element_identifier[j] = i
                    unique_els -= 1
            end
        end

        push!(unique_idxs, i)
    end
    return unique_els, element_identifier, unique_idxs  
end


function unique_identifiers_old!(exprs, terms)
    unique_idents = length(exprs)
    delete_idxs = []
    for i in eachindex(exprs)
        for j in i+1:length(exprs)
            if exprs[i] == exprs[j]
                if !isempty(terms[i])
                    push!(terms[i], j)
                    terms[j] = []
                    unique_idents -= 1
                    push!(delete_idxs, j)
                end
            end
        end
    end
    deleteat!(terms, sort!(delete_idxs))
    return unique_idents
end

"""
In an expression replace all the reserved symbols for an algorithm
Gotten from the reserved_symbols function of the algorithm
"""
function replace_reserved(algo::Type{<:MCAlgorithm}, expr::Expr)
    _reserved_symbols = reserved_symbols(algo)
    for (old, new) in _reserved_symbols
        expr = symbwalk(x -> x == old ? new : x, expr)
    end
    return expr
end

"""
In an expression with symbols of the form x_i, x_j, x_k, replace them with x[i], x[j], x[k]
"""
function replace_indices(expr)
    replacefunc = x -> (replaced = i_to_vec_expr(x); !isnothing(replaced) ? replaced : x)
    symbwalk(replacefunc, expr)
end

"""
Replace all the i's in an expression with idx
WHY?
"""
function replace_idxs(expr::Expr)
    symbwalk(x -> x == :i ? :idx : x, expr)
end


"""
Remove a key from a NamedTuple
"""
function remove_key(nt::NamedTuple, key::Symbol)
    return NamedTuple{tuple(deleteat!(collect(propertynames(nt)), findfirst(isequal(key), propertynames(nt)))...)}(
        deleteat!(collect(nt), findfirst(isequal(key), propertynames(nt)))
    )
end

function find_symb(expr, symb, index...)
    if expr isa Expr
        for (argidx,arg) in enumerate(expr.args)
            found = find_symb(arg, symb, index..., argidx)
            if found[1]
                return found
            end
        end
    elseif expr isa Symbol
        if expr == symb
            return true, index
        end
    end
    return false, nothing
end

function delete_tree(expr, indexs)
    cexpr = deepcopy(expr)
    previous_expr = cexpr
    branch = cexpr
    for i in indexs[1:end-2]
        previous_expr = branch
        branch = branch.args[i]
    end
    deleteat!(branch.args, indexs[end-1])
    # If only one branch left, remove the operator
    if length(branch.args) == 2 && branch.args[1] == :+
        previous_expr.args = branch.args[2:2]
    end
    return cexpr
end

function replace_symb(expr, val, indexs)
    cexpr = deepcopy(expr)
    branch = cexpr.args
    for i in indexs[1:end-1]
        branch = branch[i].args
    end
    branch[indexs[end]] = val
    return cexpr
end

"""
Checks ParamVals for inactivity and removes all inactive symbol branches
"""
function replace_inactive_symbs(params, expr::Expr)
    for (name,param) in zip(keys(params),params)
        if isinactive(param)
            if name == :self
                name = :self_i
            end
            found, indexs = find_symb(expr, name)
            if found
                if default(param) == 0
                    expr = delete_tree(expr, indexs)
                else
                    expr = replace_symb(expr, default(param), indexs)
                end
            end
        end
    end
    return expr
end

"""

"""
function substitute_symbols(algorithm::Type{<:MCAlgorithm}, params, expr::Expr)
    #Replace inactive symbols

    # Replace the reserved symbols
    subs_return_exprs = replace_reserved(Metropolis, Meta.parse(body))
    
    # For the remaining, get them from gparams.param[indices...]
    subs_return_exprs = replace_indices(subs_return_exprs)
end

function group_idxs(layertypes)
    grouped = []
    idxs = [1]
    lts = layertypes.parameters
    lasttype = lts[1]
    for (lt_idx,lt) in enumerate(lts)
        if equiv(lt, lasttype)
            continue
        end
        push!(grouped, first(indexset(lasttype)):last(indexset(lts[lt_idx-1])))
        push!(idxs, lt_idx)
        lasttype = lt
    end
    push!(grouped, first(indexset(lasttype)):last(indexset(lts[end])))
    return grouped, idxs
end