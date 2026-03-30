export AbstractWeightGenerator, WeightGenerator, IsingWG, @WG
export WeightGeneratorOld, @WGOld

abstract type AbstractWeightGenerator end

struct WeightGenerator{F, NN} <: AbstractWeightGenerator
    func::F
    funcexp::Union{String, Expr, Symbol, Function, Nothing}
    rng::Random.AbstractRNG
end

function WeightGenerator(func, NN = tuple(1), rng = Random.MersenneTwister(); exp = nothing)
    func = DirectMethod(func; allowedkwargs = [:dr, :c1, :c2, :dc])
    WeightGenerator{typeof(func),NN}(func, exp, rng)
end

pass_existing_kwargs(wg::WeightGenerator; kwargs...) = pass_existing_kwargs(wg.func; kwargs...)

getNN(wg::WeightGenerator{F, NN}) where {F,NN} = NN
function getNN(wg::WeightGenerator{F, NN}, dims) where {F,NN}
    if NN isa Integer
        return ntuple(i -> NN, dims)
    else
        return NN
    end
end

macro WG(func, kwargs...)
    kwargs = macro_parse_kwargs(kwargs, :NN => Union{Int, NTuple, Symbol}, :rng, NN = 1, rng = MersenneTwister())
    funcexp = QuoteNode(remove_line_number_nodes(func))
    newfunc = quote @inline WeightGenerator($func, $(kwargs[:NN]), $(kwargs[:rng]), exp = $funcexp) end

    return esc(newfunc)
end


@inline function (wg::WeightGenerator{F})(;dr::DR, c1::C1 = nothing, c2::C2 = nothing, dc::DC = nothing) where {F<:DirectMethod, DR, C1, C2, DC}
    return pass_existing_kwargs(wg;dr, c1, c2, dc)
end



function WeightGeneratorOld(func, NN = tuple(1), rng = Random.MersenneTwister(); exp = nothing)
    WeightGenerator{typeof(func),NN}(func, exp, rng)
end

macro WGOld(func, kwargs...)
    kwargs = macro_parse_kwargs(kwargs, :NN => Union{Int, NTuple, Symbol}, :rng, NN = 1, rng = MersenneTwister())

    #Function parsing
    # Either anonymous function which has to have a combination of
    # 
    # 
    # Or a global function with the same with any set of these arguments
    f_argnames = nothing
    f_location = nothing
    try 
        # Try to eval func directly
        f_argnames = method_argnames(last(methods(eval(func))))[2:end]
        f_location = :anonymous
    catch
        try # Else try to eval in Main
            f_argnames = method_argnames(last(methods(eval(:(Main.$func)))))[2:end]
            f_location = :global
        catch
            error("Could not evaluate function $func. Make sure it is defined.")
        end
    end
    # println("Function argnames are: $f_argnames")
    # Check if argnames only contain a subset of the symbols allowedargs_func
    allowedargs_func = [:dr, :c1, :c2, :dc]
    if !(all([arg ∈ allowedargs_func for arg in f_argnames]))
        error("Function must only contain arguments $allowedargs_func")
    end

    newfunc = nothing
    funcexp = nothing
    # println("Function location is: $f_location")
    if f_location == :anonymous
        funcbody = func.args[2]
        newfunc = quote @inline (dr, c1, c2, dc) -> $funcbody end
        funcexp = QuoteNode(remove_line_number_nodes(func))
    else
        newfunc = quote @inline (dr, c1, c2, dc) -> $func($(f_argnames...)) end
        funcexp = QuoteNode(remove_line_number_nodes(func))
    end
    # End of function parsing

    return esc(:(WeightGenerator($(newfunc), $(kwargs[:NN]), $(kwargs[:rng]), exp = $funcexp)))
end

"""
Old version
"""
@inline function (wg::WeightGenerator)(;dr::DR, c1::C1 = nothing, c2::C2 = nothing, dc::DC = nothing) where {DR, C1, C2, DC}
    # return @inline wg.func(dr)
    if @inline hasmethod(wg.func, Tuple{DR, C1, C2, DC})
        return @inline wg.func(dr, c1, c2, dc)
    elseif @inline hasmethod(wg.func, Tuple{DR, C1, C2})
        return @inline wg.func(dr, c1, c2)
    elseif @inline hasmethod(wg.func, Tuple{DR, DC})
        return @inline wg.func(dr, dc)
    elseif @inline hasmethod(wg.func, Tuple{DR})
        return @inline wg.func(dr)
    else
        error("Function does not have a method for the given argument types. Make sure it is defined with the correct argument names.")
    end
end

const IsingWG = @WG dr -> dr == 1 ? 1 : 0 NN=1











