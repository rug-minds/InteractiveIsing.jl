# TODO: Use prefs
debugmode = true
@static if debugmode
export strsplice, find_i_symb, symbol_without_index, match_indexed_symbol, 
    before_underscore, i_to_vec_expr, symbwalk, _symbwalk, identify_unique_elements, 
    unique_identifiers_old!, replace_reserved, substitute_zero, substitute_one, replace_indices, 
    replace_idxs, remove_key, replace_inactive_symbs, symbol_without_index, find_symb, find_symbs, delete_tree
end

function same_dispatch(layertype1, layertype2)
    if layertype1.parameters[1] == layertype2.parameters[1] && layertype1.parameters[2] == layertype2.parameters[2]
        return true
    end
    return false
end

function grouped_idxs(unshuffled_layers_type)
    layertypes = unshuffled_layers_type.parameters
    grouped_idxs = []
    for layer_idx in eachindex(layertypes)
        idxset = layerparams(layertypes[layer_idx], Val(:IndexSet))
        if layer_idx == 1
            push!(grouped_idxs, idxset[1])
            push!(grouped_idxs, idxset[end])
            continue
        end
        
        if same_dispatch(layertypes[layer_idx], layertypes[layer_idx-1])
            idxset[end] = last(grouped_idxs[end])
        else
            push!(grouped_idxs, idxset[1])
            push!(grouped_idxs, idxset[end])
        end

    end
    # Make ranges
    ranges = []
    for i_idx in 1:2:length(grouped_idxs)
        push!(ranges, grouped_idxs[i_idx]:grouped_idxs[i_idx+1])
    end
    return ranges
end

"""
Switches algorithm based on layertype
Inlined so no runtime dispatch
"""
# function layerswitch(@specialize(func), i, layers::LayerTuple, @specialize(args)) where LayerTuple
#     idxsets = tuple_type_property(indexset, layers)
#     @inline _layerswitch(func, i , 1, gethead(layers), gethead(idxsets), gettail(layers), gettail(idxsets), args)
# end

# function _layerswitch(func, i, layeridx, thislayer, thisidxset, layerstail, setstail, args)
#     if i in thisidxset
#         return @inline func(i, args, layeridx, thislayer)
#     end
#     return @inline _layerswitch(func, i, layeridx+1, gethead(layerstail), gethead(setstail), gettail(layerstail), gettail(setstail), args)
# end

# function _layerswitch(::Any, ::Any, ::Any, ::Nothing, ::Any, ::Any, ::Any, ::Any)
#     throw(BoundsError("Index out of bounds"))
# end
function layerswitch(@specialize(func), i, layers::LA, @specialize(args)) where LA
    idxsets = @inline tuple_type_property(indexset, layers.metadatas)
    @inline _layerswitch(func, i , 1, gethead(layers.metadatas), gethead(idxsets), gettail(layers.metadatas), gettail(idxsets), args)
end

function _layerswitch(func, i, layeridx, thislayermd, thisidxset, layersmd_tail, sets_tail, args)
    if i in thisidxset
        return @inline func(i, args, layeridx, thislayermd)
    end
    return @inline _layerswitch(func, i, layeridx+1, gethead(layersmd_tail), gethead(sets_tail), gettail(layersmd_tail), gettail(sets_tail), args)
end

function _layerswitch(::Any, ::Any, ::Any, ::Nothing, ::Any, ::Any, ::Any, ::Any)
    throw(BoundsError("Index out of bounds"))
end


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
In a symbol, find the pattern _(combination of indexes)
Returns the range of indexes and the symbols of the (i,j,k,l,m,n) found
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
        while (found_last_symb+1 <= lastindex(str)) && (str[found_last_symb+1] in ['i','j','k','l','m','n'])
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
Find all characters in a symbol before the underscore
"""
function before_underscore(symb)
    sstring = String(symb)
    if sstring[1] == '_'
        return nothing, nothing
    end
    lastidx = nothing
    for idx in eachindex(sstring)
        if sstring[idx] == '_'
            break
        end
        lastidx = idx
    end
    if isnothing(lastidx)
        return nothing, nothing
    end

    return Symbol(sstring[1:lastidx]), 1:lastidx
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
In an expression containing some struct and fields, like s.field1, s.field2, s.field3
    find all the idxs where the struct is and the name of the field being acessed
"""
function find_struct_field_access(expr, structname)
    idxss = find_symbs(expr, structname)
    hits = []
    for idxs in idxss
        if get_head(expr, idxs[1:end-1]...) == :.
            fieldname = enter_args(expr, idxs[1:end-1]...,2)
            push!(hits, (idxs, fieldname))            
        end
    end
    if isempty(hits)
        return nothing
    end
    return hits
end
export find_struct_field_access





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

"""
Take the symbols that are reserved by thing and replace them in the expression
"""
function replace_reserved!(exp::Expr, thing)
    symbols = reserved_symbols(thing)
    for (old, new) in symbols
        exp = symbwalk(x -> x == old ? new : x, exp)
    end
    return exp
end

"""
In an expression replace all the reserved symbols for an algorithm
Gotten from the reserved_symbols function of the algorithm
"""
function replace_reserved(algo::Type, expr::Expr)
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

"""
For a symbol in the form x_ijk return x
"""
function symbol_without_index(symb)
    idxs, _ = find_i_symb(symb)
    if isnothing(idxs)
        return symb
    end
    newstr = String(symb)
    str = newstr[1:first(idxs)-1]
    return Symbol(str)
end

function match_indexed_symbol(symb1,symb2)
    before_underscore(symb1)[1] == before_underscore(symb2)[1]
end

"""
In an expression, find a symbol
Will match with any symbol that is a prefix of the symbol, e.g. x matches x_ijk
"""
function find_symb(expr, symb, index...)
    if expr isa Expr
        for (argidx,arg) in enumerate(expr.args)
            found = find_symb(arg, symb, index..., argidx)
            if !isnothing(found)
                return found
            end
        end
    elseif expr isa Symbol
        # if expr == symb
        if match_indexed_symbol(symb, expr)
            return index
        end
    end
    return nothing
end

"""
In an expression find all positions of a symbol
Will match with any symbol that is a prefix of the symbol, e.g. x matches x_ijk
Returns a list of indices for the consecutive levels of the args of an expression
"""
function find_symbs(expr, symb, all_found = [], index...)
    if expr isa Expr
        for (argidx,arg) in enumerate(expr.args)
            find_symbs(arg, symb, all_found, index..., argidx)
        end
    elseif expr isa Symbol
        # if expr == symb
        if match_indexed_symbol(symb, expr)
            push!(all_found, index)
            return all_found
        end
    end
    return all_found
end

enter_args(ex::Expr, t::Tuple) = enter_args(ex, t...)
function enter_args(ex::Union{Expr,QuoteNode}, idxs::Int...)
    if length(idxs) == 0
        return ex
    else
        return enter_args(ex.args[first(idxs)], idxs[2:end]...)
    end
end

function enter_args(ex::Symbol, idxs...)
    if length(idxs) == 0
        return ex
    else
        throw(ArgumentError("type Symbol has no field args"))
    end
end
function enter_args(ex::LineNumberNode, idxs...)
    if length(idxs) == 0
        return ex
    else
        throw(ArgumentError("type LineNumberNode has no field args"))
    end
end

function get_args(a::Expr,b...)
    ex = enter_args(a,b...)
    if ex isa Symbol
        return ex
    end

    return ex.args
end

get_head(a,b...) = enter_args(a,b...).head
delete_branch(ex::Expr, idxs::Int...) = deleteat!(enter_args(ex, idxs[1:end-1]...).args, idxs[end])
delete_branch(a,b::Tuple) = delete_branch(a,b...)

export enter_args, delete_branch, get_args, get_head

"""
In a mathematical expression subsitute a symbol with zero
Indexes give the path to the symbol in the following way, expr.args[indexs[1]].args[indexs[2]]...args[indexs[end]]
It will delete all the branches including the symbol that completely evaluate to zero
"""
function substitute_zero(expr, indexs)
    cexpr = deepcopy(expr)
    if indexs isa Integer || length(indexs) == 1
        cexpr = Expr(:block, cexpr)
        indexs = (1, indexs...)
    end

    # Walk up the branch and fix the tree
    for level in 1:length(indexs)
    # for level in 1:length(indexs)
        this_level = length(indexs)-level
        level_indexs = indexs[1:this_level]
        level_indexs = indexs[1:this_level]
        previous_branch = indexs[this_level+1]

        symbol = get_args(cexpr, level_indexs...)[1]

        # If the whole branch evaluates to zero, go to the next
            # if multiplication
            if symbol == :*
                continue
            end

            # If division and its the numerator
            if symbol == :/ && level_indexs[end] == 2
                continue
            end

            # If a power
            if symbol == :^
                continue
            end

        # Delete the zero
        deleteat!(get_args(cexpr, level_indexs...), previous_branch)

        # Fix the operator
        if (symbol == :+ || symbol == :-)
            if length(get_args(cexpr, level_indexs...)) <= 2
                # replace branch with leftover branch

                leftover = get_args(cexpr, level_indexs...)[2]
                get_args(cexpr, indexs[1:this_level-1]...)[indexs[this_level]] = leftover
            end
        end
    
        break
    end

    return cexpr
end

"""
In a mathematical expression subsitute a symbol with one
Indexes give the path to the symbol in the following way, expr.args[indexs[1]].args[indexs[2]]...args[indexs[end]]
It will delete all the branches including the symbol that completely evaluate to one
"""
function substitute_one(expr, indexs)
    cexpr = deepcopy(expr)

    # Walk up the branch and fix the tree
    for level in 1:length(indexs)-1
        this_level = length(indexs)-level
        level_indexs = indexs[1:this_level]
        previous_branch = indexs[this_level+1]

        symbol = get_args(cexpr, level_indexs...)[1]
        
        # If a power of 1, means next branch also evaluates to 1
        if symbol == :^
            continue
        end

        deleteat!(get_args(cexpr, level_indexs...), previous_branch)

        # Then fix the symbol by replacing the whole branch with the leftover
        if (symbol == :* || symbol == :/)
            if length(get_args(cexpr, level_indexs...)) == 2
                leftover = get_args(cexpr, level_indexs...)[2]
                get_args(cexpr, indexs[1:this_level-1]...)[indexs[this_level]] = leftover
            end
        end
        break
    end
    return cexpr
end

"""
Replace a symbol in an expression with a value
"""
function replace_symb(expr, val, indexs)
    cexpr = deepcopy(expr)
    if indexs isa Integer || length(indexs) == 1
        if !(indexs isa Integer)
            indexs = indexs[1]
        end

        cexpr.args[indexs] = val
        return cexpr
    end
    branch = cexpr.args
    for i in indexs[1:end-1]
        branch = branch[i].args
    end
    branch[indexs[end]] = val
    return cexpr
end
export replace_symb

"""
Replace a symbol in an expression with a value
    If the value is 0, it will delete the branch
    If the value is 1, it will delete the branch if it is a multiplication or division
"""
function substitute_math(expr, val, indexs)
    if val == 0
        return substitute_zero(expr, indexs)
    elseif val == 1
        return substitute_one(expr, indexs)
    else
        return replace_symb(expr, val, indexs)
    end
end
export substitute_math

"""
Checks ParamVals for inactivity and removes all inactive symbol branches
"""
function replace_inactive_symbs(params, expr::Expr)
    for (name,param) in zip(keys(params),params)
        if isinactive(param)
            while (indexs = find_symb(expr, name); !isnothing(indexs))
                    if default(param) == 0
                        expr = substitute_zero(expr, indexs)
                    elseif default(param) == 1
                        expr = substitute_one(expr, indexs)
                    else
                        expr = replace_symb(expr, default(param), indexs)
                    end
            end
        end
    end
    return expr
end
"""
Replace the parameters with g.params._nt[param] in the expression
"""
function replace_params(params, expr::Expr) 
    for symb in keys(params)
        while (indexs = find_symb(expr, symb); !isnothing(indexs))
            newsymb = :(gparams.$symb)
            expr = replace_symb(expr, newsymb, indexs)
        end
    end
    replace_inactive_symbs(params, expr)
end

"""
For a function that uses ParamVals, replace the symbols that are inactive by their default values
    als point them to the gparams.param[indices...]
"""
function substitute_symbols(algorithm::Type{T}, params, expr::Expr) where T
    #Replace inactive symbols

    # Replace the reserved symbols
    subs_return_exprs = replace_reserved(T, Meta.parse(body))
    
    # For the remaining, get them from gparams.param[indices...]
    subs_return_exprs = replace_indices(subs_return_exprs)

end


"""
For a function that uses ParamVals, replace the symbols that are inactive by their default values
"""
function param_function(func_expr, Algo, params)
    # Replace the symbols that are inactive by their default values
    H_ex = replace_inactive_symbs(params, func_expr)
    # Replace the symbols that are reserved by the algorithm
    H_ex = replace_reserved(Algo, H_ex)
    # Replace the symbols that are indexed by their vector form
    H_ex = replace_indices(H_ex)
    # Replace the symbols that are parameters by gparams.param
    H_ex = replace_params(params, H_ex)
    return H_ex
end