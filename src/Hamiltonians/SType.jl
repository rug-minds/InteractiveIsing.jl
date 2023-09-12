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

# This is written in a macro to make it easy to add dispatch types later
#   E.g. if we want to add a continuous/discrete dispatch type, we can just add it to the macro
#   and it will be added to all the functions that use it
macro sfactors(factors...)
    fullexpr = quote end
    str = "const factors = ["
    str *= ':'*join(factors, ", :") *']'
    type_parameter_names = (join(factors, ','))
    # str = "const factors = [factors...]"
    push!(fullexpr.args, Meta.parse(str))

    push!(fullexpr.args, Meta.parse("struct SType{$type_parameter_names} end "))
    push!(fullexpr.args, Meta.parse(
        "function changeSParam(type::SType{$type_parameter_names}, pairs::Pair...) where {$type_parameter_names}
            newparams = [typeof(type).parameters...]
            for pair in pairs
                newparams[findfirst(x->x==pair.first, factors)] = pair.second
            end
            return SType{newparams...}()
        end"
    ))

    push!(fullexpr.args, Meta.parse("export SType"))
    # push!(fullexpr.args, Meta.parse(
    #     "function getSParam(type::Type{SType{$type_parameter_names}}, sym::Symbol) where {$type_parameter_names}
    #         return type.parameters[findfirst(x->x==sym, factors)]
    #     end"))

    return esc(fullexpr)
end

# Define the factors and SType struct
@sfactors Weighted Magfield Clamp Defects
"""
Default SType
"""
SType() = SType{true, false, false, false}()

"""
Get the parameter of the SType that matches the given symbol
"""
function getSParam(type::SType, sym::Symbol)
    return typeof(type).parameters[findfirst(x->x==sym, factors)]
end
getSParam(type::Type{ST}, sym::Symbol) where {ST <: SType} = getSParam(ST(), sym)

"""
Get a modified version of the standard SType where the given parameters are changed
"""
function SType(pairs::Pair...)
    return changeSParam(SType(), pairs...)
end

function Base.show(io::IO, st::SType) 
    println(io, "SType with parameters:")
    for (idx, factor) in enumerate(string.(factors))
        print(io, " $factor => $(typeof(st).parameters[idx])")
        if idx != length(factors)
            print(io, "\n")
        end
    end
end