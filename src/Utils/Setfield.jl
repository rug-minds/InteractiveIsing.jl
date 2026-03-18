"""
General replacing setfield
"""
@inline setfield(s::S, name::Symbol, val::V) where {S,V} = @inline setfield(s, Val(name), val)
@inline @generated function setfield(s::S, name::Val{FieldName}, val::V) where {S,V, FieldName}
    fieldnames = Base.fieldnames(S)
    field_match = findfirst(==(FieldName), fieldnames)
    if isnothing(field_match)
        error("Field $(FieldName) not found in struct $(S)\nfieldnames are: $(fieldnames)\nlooking for field of type $(V)")
    end
    parameters = S.parameters
    constructor_ref = GlobalRef(parentmodule(S), nameof(S))
    parameter_match = findfirst(==(fieldtype(S, FieldName)), tuple(parameters...))
    return_type_expr = :($S)
    if !isempty(parameters) && !isnothing(parameter_match) && !(S <: NamedTuple)
        parameters = (parameters[1:(parameter_match - 1)]..., val, parameters[(parameter_match + 1):end]...)
        parameters = map(x -> x isa Symbol ? QuoteNode(x) : x, parameters)

        return_type_expr = Expr(:curly, constructor_ref, parameters...)
    end    

    getfields = Any[:(getfield(s, $(QuoteNode(field)))) for field in fieldnames]
    getfields[field_match] = :(val)

    #Namedtuple handling
    # If it's a namedtuple, we only need the names
    if S <: NamedTuple
        parameters = tuple(S.parameters[1]...)
        return_type_expr = Expr(:curly, :($(nameof(S))), parameters)
        getfields = tuple(Expr(:tuple, getfields...))
    end

    
    exp = Expr(:call, constructor_ref, getfields...)
    # error("Exp: $exp")


    ### ERROR:
        exp_str = sprint(show, exp)
        type_expr_str = sprint(show, return_type_expr)
        getfields_str = repr(getfields)
        msg = string(
            "\n--- setfield debug ---\n",
            "S = ", S, "\n",
            "FieldName = ", FieldName, "\n",
            "V = ", V, "\n\n",
            "fieldnames = ", fieldnames, "\n",
            "field_match = ", field_match, " (", fieldnames[field_match], ")\n\n",
            "S.parameters = ", S.parameters, "\n",
            "parameter_match = ", parameter_match, "\n",
            "new parameters = ", parameters, "\n\n",
            "type_expr = ", type_expr_str, "\n",
            "getfields = ", getfields_str, "\n",
            "exp = ", exp_str, "\n",
        )

    final_struct_expr = Expr(:(::), exp, return_type_expr)
    return quote
        try
            # $exp
            $(final_struct_expr)
        catch e
            error($msg * "\nOriginal error: " * sprint(showerror, e))
        end
    end
    # return exp
end

@inline function setfields(s::S, names::Tuple, vals...) where {S}
    final = unrollreplace(s, Val(length(vals))) do current, idx
        setfield(current, Val(names[idx]), vals[idx])
    end
    return final
end
