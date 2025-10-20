export WeightGenerator, IsingWG, @WG
struct WeightGenerator{F, NN}
    func::F
    funcexp::Union{String, Expr, Symbol, Function, Nothing}
    rng::Random.AbstractRNG
end

getNN(wg::WeightGenerator{F, NN}) where {F,NN} = NN
getNN(wg::WeightGenerator{F, NN}, dims) where {F,NN} = ntuple(i -> (i <= length(NN) ? NN[i] : 1), Val(length(dims)))

function WeightGenerator(func, NN = tuple(1), rng = Random.MersenneTwister(); exp = nothing)
    WeightGenerator{typeof(func),NN}(func, exp, rng)
end

macro WG(func, kwargs...)
    kwargs = macro_parse_kwargs(kwargs, :NN => Union{Int, NTuple}, :rng, NN = 1, rng = MersenneTwister())

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
            f_argnames = method_argnames(last(methods(eval(:($func)))))[2:end]
            f_location = :global
        catch
            error("Could not evaluate function $func. Make sure it is defined.")
        end
    end
    # println("Function argnames are: $f_argnames")
    # Check if argnames only contain a subset of the symbols allowedargs_func
    allowedargs_func = [:dr, :c1, :c2]
    if !(all([arg ∈ allowedargs_func for arg in f_argnames]))
        error("Function must only contain arguments $allowedargs_func")
    end

    newfunc = nothing
    funcexp = nothing
    # println("Function location is: $f_location")
    if f_location == :anonymous
        funcbody = func.args[2]
        newfunc = quote @inline (dr, c1, c2) -> $funcbody end
        funcexp = QuoteNode(remove_line_number_nodes(func))
    else
        newfunc = quote @inline (dr, c1, c2) -> $func($(f_argnames...)) end
        funcexp = QuoteNode(remove_line_number_nodes(func))
    end
    # End of function parsing

    return esc(:(WeightGenerator($(newfunc), $(kwargs[:NN]), $(kwargs[:rng]), exp = $funcexp)))
end


function (wg::WeightGenerator)(;dr, c1 = nothing, c2 = nothing)
    return @inline wg.func(dr, c1, c2)
end

const IsingWG = @WG dr -> dr == 1 ? 1 : 0 NN=1















