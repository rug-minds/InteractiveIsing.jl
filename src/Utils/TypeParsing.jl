"""
Args from 1 to last_parsed_idx are parsed
    Return the rest
"""
remove_parsed_args(args, last_parsed_idx::Int) = args[(last_parsed_idx+1):end]
remove_parsed_args(args, ::Nothing) = error("Attempted to remove parsed args, but no args were parsed.")

"""
Remove an optional parsed argument from the args tuple. If the parsed_idx is nothing, return the original args.
"""
function remove_optional_parsed_arg(args, parsed_idx)
    isnothing(parsed_idx) ? args : args[setdiff(1:length(args), parsed_idx)]
end

"""
From a type and a args..., find the first element of the given type in the args, return it and the rest of the args without it.
    If no element of the given type is found, return default and the original args (or throw an error if error = true and default = nothing)
"""
function type_parse(type, args...; default = nothing, error = true)
    t_idx = findfirst(x -> x isa type, args)
    if isnothing(t_idx)
        if error && isnothing(default)
            error("Expected argument of type $type not found in arguments: $args")
        else
            return default, args
        end
    else
        el = args[t_idx]
        args = remove_optional_parsed_arg(args, t_idx)
        return el, args
    end
end

function parse_by_func(func, args...; default = nothing, error = true)
    t_idx = findfirst(x -> func(x), args)
    if isnothing(t_idx)
        if error && isnothing(default)
            error("Expected argument matching function $func not found in arguments: $args")
        else
            return default, args
        end
    else
        el = args[t_idx]
        args = remove_optional_parsed_arg(args, t_idx)
        return el, args
    end
end