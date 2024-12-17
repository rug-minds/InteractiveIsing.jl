"""

"""
const allHMacros = Dict{Type, Dict{Symbol, <:Function}}()

function H_Macros(::T, pairs::Pair{Symbol, <:Function}...) where T
    if !haskey(allHMacros, T)
        allHMacros[T] = Dict(pairs...)
    else
        for (k, v) in pairs
            allHMacros[T][k] = v
        end
    end
end

function H_Macros(::T, symbol::Symbol) where T
    return allHMacros[Type][symbol]
end

function is_registered_HMacro(::Type, ex) 
    return ex.head == :macrocall && haskey(allHMacros[Type], ex.args[1])
end

function indexed_position(symb, index_symb = nothing)
    symbstring = string(symb)
    idx = findfirst(x == '_', symbstring)

    # If index_symb is given, check if the index symbol after the underscore matches
    if !isnothing(index_symb)
        if symbstring[idx+1:end] == string(index_symb)
            return idx
        else
            return nothing
        end
    end
end

function replace_symb_index!(symb, num, index_symb = nothing)
    idx = indexed_position(symb, index_symb)
    if !isnothing(idx)
        ssymbol = string(symb)
        splice!(ssymbol, idx:idx, string(num))
        return Symbol(ssymbol)
    end
end

function iterate_exp(hams, ex, symbol = :i)
    finalexp = Expr(:block)
    for hidx in 1:length(hams)
        ham = hams[hidx]
        if is_registered_HMacro(ham, ex)
            push!(finalexp.args, H_Macros(ham, symbol)(hams))
        end
    end
end

function symbol_indexes(symb::Symbol)
    symbstring = string(symb)
    idx = findfirst(x == '_', symbstring)
    if !isnothing(idx)
        return [symbstring[idx+1:end]...]
    end
    return nothing
end

function find_contractions(exp, c_symb)
    contractions = []
    
    return contractions
end

function _find_contractions(exp, c_symb, contractions, idxs = [])
    args = get_args(exp, idxs)
    if args[1] == :*
        types = typeof.(args[2:end])
        has_Symbol = findall(x -> x == Symbol, types)
        if !isempty(has_Symbol)
            
        end
    end
end

