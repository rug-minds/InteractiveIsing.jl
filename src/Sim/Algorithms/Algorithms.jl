
include("EnergyFuncs/EnergyFuncs.jl")

include("Metropolis.jl")
include("Langevin.jl")

include("LayeredAlgorithms.jl/LayeredAlgorithms.jl")

export prepare

"""
Fallback preparation for updateFunc
"""
# _prepare(algorithm::Any, ::Any; kwargs...) = error("No prepare function defined for $(typeof(algorithm))")

let times = Ref([])
    global function upDebug(g, params, lTemp, gstate::Vector, gadj, iterator, rng, gstype, dEFunc)

        β = 1/(lTemp[])
        
        idx = rand(rng, iterator)
        
        ti = time()
        Estate = @inbounds gstate[idx]*dEFunc(g, gstate, gadj, idx, gstype)
        tf = time()

        push!(times[], tf-ti)
        if length(times[]) == 1000000
            println(sum(times[])/length(times[]))
            times[] = []
        end

        minEdiff = 2*Estate

        if (Estate >= 0 || rand(rng) < exp(β*minEdiff))
            @inbounds g.state[idx] *= -1
        end
        
    end
end
export upDebug



"""
Get a string of the arguments from a defined get_args function that returns
    a tuple of symbols that represent the arguments for a function
"""
function get_args_string(func_type, addbrackets = nothing)
    args = "$(get_args(func_type))"
    args = "(;"*args[2:end]
    # Remove colons
    args = replace(args, ":" => "")
    if !isnothing(addbrackets)
        args = args[1:end-1]
        args *= ", $addbrackets)"
    end
    return args
end

"""
Get the expression for a generated algorithm
"""
function getexpression(f::Function, g)
    args = prepare(f,g )
    println("Expression for function $(f) \n")
    exp = gen_exp(f, args)
    println(exp)
    println()

    return exp
end

export getexpression



## Iterator that changes without having to recompile the loop
struct DynamicalIterator{GT}
    g::GT
end

function Base.rand(rng::MersenneTwister, it::DynamicalIterator)
    return rand(rng, it.g.defects.aliveList)
end