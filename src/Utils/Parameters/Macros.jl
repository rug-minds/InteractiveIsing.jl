function find_type_in_args(args, type::Symbol)
    for arg in args
        if arg isa Expr && arg.head == :(::)
            if  arg.args[2] == type
                return arg.args[1]
            end
        end
    end
    return nothing
end

function get_field_type(::Type{T}, field::Val{S}) where {T <: NamedTuple,S}
    names = fieldnames(T)
    types = fieldtypes(T)
    
    index = findfirst(==(S), names)
    
    return index === nothing ? throw(ArgumentError("Field $field not found")) : types[index]
end

function get_field_type(@specialize(nt::NamedTuple), field)
    return get_field_type(typeof(nt), field)
end

get_field_type(nt, field::Symbol) = get_field_type(nt, Val{field}())

get_field_type(ip::Parameters{nt}, field::Symbol) where nt = get_field_type(nt, field)

function replace_struct_field_access(func, paramname, replace)
    find_struct_accesses = find_struct_field_access(func, paramname)
    quotes = []
    places = []
    for access in find_struct_accesses
        idxs = access[1]
        fieldname = access[2]
        q = replace(func, paramname, fieldname, idxs)
        push!(quotes, q)
        push!(places, idxs[1:end-2])
    end
    for (qidx,q) in enumerate(quotes)
        func = replace_symb(func, q, places[qidx])
    end
    return func
end

function replace_paramval(func, paramname)
    function replacefunc(containing_exp, paramname, fieldname) 
        containing_exp = enter_args(func, idxs[1:end-2])
        if containing_exp.head == :ref
            replaced = :(default($paramname, $fieldname))
        else
            replaced = replace_symb(containing_exp, :(default($paramname, $fieldname)), idxs[end-1])
        end
        q = quote   if isactive($paramname, $(fieldname))
                    $(containing_exp)
                else
                    $(replaced)
                end
        end
        return q
    end
    replace_struct_field_access(func, paramname, replacefunc)
end
using LoopVectorization

"""
Find all places in an expression where we have str.field or str.field[idx]
"""
function find_struct_access(exp, symb)
    find_idxs = find_symbs(exp, symb)
    deletes = []
    for i_idx in eachindex(find_idxs)
        this_idxs = find_idxs[i_idx][1:end-1]
        new_idxs = this_idxs
        head = get_head(exp, this_idxs)
        if head != :. && head != :ref
            push!(deletes, i_idx)
            continue
        end
        while (head == :ref || head == :.)
            new_idxs = new_idxs[1:end-1]
            head = get_head(exp, new_idxs)
        end
        find_idxs[i_idx] = find_idxs[i_idx][1:(length(new_idxs)+1)]
    end
    deleteat!(find_idxs, deletes)

    return find_idxs
end



"""
Replace all struct accesses of a symbol in an expression with a new expression
    Give a function to replace the expression in the form replace(this_exp)
"""
function replace_struct_access(exp, symb, replace)
    find_idxs = find_struct_access(exp, symb)
    for i_idx in eachindex(find_idxs)
        this_idxs = find_idxs[i_idx]
        this_exp = enter_args(exp, this_idxs)
        this_replace = if replace isa Function
                        replace(this_exp)
                    else
                        replace
                    end
            
        replace_args!(exp, this_idxs, this_replace)
    end
end

function structname_fieldname(struct_access_exp)
    found = (expmatch(struct_access_exp) do head, args
        if head == :.
            return true
        end
        return false
    end)
    structname = found[1].args[1]
    fieldname = found[1].args[2]
    return structname, fieldname
end

function default_paramval(paramname, fieldname)
    :(default($paramname, $fieldname))
end

function ifactive_defaultparamval(exp, paramname, fieldname)
    quote   if isactive($paramname, $(fieldname))
                $exp
            else
                default($paramname, $fieldname)
            end
    end
end

"""
From a list of arguments with possible type annotations, 
return a list of arguments without type annotations
"""
function arguments_to_pass(a...)
    new_a = []
    for arg in a
        if arg isa Expr && arg.head == :(::)
            push!(new_a, arg.args[1])
        else
            push!(new_a, arg)
        end
    end
    return new_a
end

function inline_default(exp, params::Union{Parameters, Type{<:Parameters}}, fieldname)
    # fieldname = Symbol(fieldname)
    if fieldname isa QuoteNode
        fieldname = fieldname.value
    end

    return if isactive(params, fieldname)
        exp
    else
        default(params, fieldname)
    end
end


# """
# """
# macro Parameters(func)
#     # println("Func ", func)
#     @capture(func, function name_(a__) body__ end)
#     paramname = find_type_in_args(a, :Parameters)
#     replace_struct_access(func, paramname, (exp) -> ifactive_defaultparamval(exp, structname_fieldname(exp)...))
#     println(func)
#     return esc(func)
# end

function matchfunction(funcexp)
    InteractiveIsing.MacroTools.@capture(funcexp, function fname_(a__) body_ end)
    if isnothing(body)
        InteractiveIsing.MacroTools.@capture(funcexp, function fname_(a__) where T_ body_ end)
        if isnothing(body)
            InteractiveIsing.MacroTools.@capture(funcexp, function fname_(a__) where {T__} body_ end)
        end
    end
    println("Matched function $fname")
    println("With args $a")
    println("With body $body")
    return fname, a, body
end

function GeneratedParametersExp(fname, body, paramname, args, replacements!::Function... = (rawbody, paramname; args...) -> rawbody)
    # wrappedbody = Expr(:quote, InteractiveIsing.remove_line_number_nodes!(body))
    wrappedbody = Expr(:quote, body)
    wrappedparamname = Expr(:quote, paramname)
    println("Wrapped body is: ")
    println(wrappedbody)
    args2pass = InteractiveIsing.arguments_to_pass(args...)
    expfunc = 
        quote 
            # Create the expression function for the generated function
            function $(Symbol(:($fname), :_exp))($(args2pass...))
                rawbody = $wrappedbody
                paramname = $wrappedparamname

                # Inline the default values
                InteractiveIsing.replace_struct_access(rawbody, paramname, (exp) -> InteractiveIsing.inline_default(exp, $(paramname), InteractiveIsing.structname_fieldname(exp)[2]))
                
                for replacement! in $(replacements!)
                    println("Replacing with $replacement!")
                    replacement!(rawbody, paramname; $(args2pass...))
                end
                return rawbody
            end
        end
    println("Expression for $fname is: ")
    printexp(expfunc)
    return expfunc
end

function GeneratedParametersGen(fname, body, args)
    args2pass = InteractiveIsing.arguments_to_pass(args...)
    func = Expr(:function, get_function_line(body))
    body = quote $(Symbol(:($fname), :_exp))($(args2pass...)) end
    push!(func.args, body)
    finalexp = quote @generated $func end
    # println("Generated function for $fname is: ")
    # println(finalexp)
    return finalexp
end

macro GeneratedParameters(func)
    fname, a, body = InteractiveIsing.matchfunction(func)
    
    paramname = InteractiveIsing.find_type_in_args(a, :Parameters)

    expfunc = InteractiveIsing.GeneratedParametersExp(fname, body, paramname, a)
    realfunc = InteractiveIsing.GeneratedParametersGen(fname, body, a)

    returnexp = quote
        $expfunc
        $realfunc
        export $fname
    end
 
    return esc(returnexp)
end
export @GeneratedParameters

