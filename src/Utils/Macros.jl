

struct DefaultAndTypePair
    default::Any
    type::Type
    DefaultAndTypePair(default = nothing , type = Any) = new(default, type)
end

getdefault(pair::DefaultAndTypePair) = pair.default
gettype(pair::DefaultAndTypePair) = pair.type

function getnt(nt, field, default = nothing)
    if haskey(nt, field)
        return nt[field]
    else
        return default
    end
end
"""
Parses keyword arguments in a macro call.
Give a list of symbols or pairs symbol=>type to specify which keyword arguments are allowed.
Defaults are passed as a keyword argument with the same name.
"""
function macro_parse_kwargs(kwargs, symbs_and_pairs::Union{Symbol, Pair}...; defaults...)
    completed_pairs = [x isa Symbol ? 
                            x => DefaultAndTypePair(getnt(defaults, x, nothing)) : 
                            x.first => DefaultAndTypePair(getnt(defaults, x.first, nothing), x.second)
                            for x in symbs_and_pairs]
    macro_parse_kwargs(kwargs, Dict{Symbol, DefaultAndTypePair}(completed_pairs...))
end

function mandatory_args(available_names) 
    [symb for (symb, dtpair) in available_names if isnothing(getdefault(dtpair))]
end


function macro_parse_kwargs(kwargs, available_names::Dict)
    key_vals = Pair[]
    leftover_names = collect(keys(available_names))
    # _mandatory_args = mandatory_args(available_names)
    
    params = prunekwargs(kwargs...)
    # println("params: ", params)
    for exp in params
        args = exp.args
        this_arg_name = args[1]
        # Delete from mandatory args if found
        deleteat!(leftover_names, findfirst(==(this_arg_name), leftover_names))

        this_arg_val = args[2]
        if haskey(available_names, this_arg_name)
            expected_type = gettype(available_names[this_arg_name])
            if eval(this_arg_val) isa expected_type || this_arg_val isa Symbol
                push!(key_vals, this_arg_name => this_arg_val)
            else
                error("Keyword argument $this_arg_name must be of type $expected_type, got $(typeof(this_arg_val))")
            end
        else
            error("Unknown keyword argument $this_arg_name")
        end
    end

    for name in leftover_names
        default = getdefault(available_names[name])
        if isnothing(default)
            error("Missing mandatory keyword argument: $name")
        else
            push!(key_vals, name => default)
        end
    end

    return (;key_vals...)
end