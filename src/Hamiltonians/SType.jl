# The SimType
# This is used for dispatching to get the right functions for the simulation type
#   E.g. used for picking a different energy function when magnetic fields are present
#   If they're not present, the magnetic field list doesn't have to be checked, which is faster

# Defines:
# const factors = [factors...]
# struct SType{factors...} end
# function SType(pairs::Pair{Symbol,T}...) where {factors..., T <: Any}
# function changeSParam(type::Type{SType{factors...}}, pairs::Pair{Symbol,T}...) where {factors..., T <: Any}
#    This is used for taking an stype and changing one or multiple of its parameters
#    Input should be a set of pairs that say which parameter needs to be changed, and what it should be changed to  
macro sfactors(factors...)
    fullexpr = quote end
    str = "const factors = ["
    str *= ':'*join(factors, ", :") *']'
    type_parameter_names = (join(factors, ','))
    # str = "const factors = [factors...]"
    push!(fullexpr.args, Meta.parse(str))

    push!(fullexpr.args, Meta.parse("struct SType{$type_parameter_names} end "))
    push!(fullexpr.args, Meta.parse(
        "function changeSParam(type::Type{SType{$type_parameter_names}}, pairs::Pair...) where {$type_parameter_names}
            newparams = [type.parameters...]
            for pair in pairs
                newparams[findfirst(x->x==pair.first, factors)] = pair.second
            end
            return SType{newparams...}
        end"
    ))

    push!(fullexpr.args, Meta.parse("export SType"))
    # push!(fullexpr.args, Meta.parse(
    #     "function getSParam(type::Type{SType{$type_parameter_names}}, sym::Symbol) where {$type_parameter_names}
    #         return type.parameters[findfirst(x->x==sym, factors)]
    #     end"))

    return esc(fullexpr)
end

@sfactors Weighted Magfield Clamp Defects Continuous

SType() = SType{true, false, false, false, :Discrete}

function getSParam(type::Type{SType}, sym::Symbol)
    return type.parameters[findfirst(x->x==sym, factors)]
end
function SType(pairs::Pair...)
    return changeSParam(SType(), pairs...)
end
