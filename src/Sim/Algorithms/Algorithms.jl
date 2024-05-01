
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


### SAMPLING
@inline sample_from_stateset(::Any, stateset::AbstractVector) = rand(stateset)
@inline sample_from_stateset(::Type{Discrete}, stateset::Tuple) = rand(stateset)
@inline sample_from_stateset(::Type{Continuous}, stateset::Tuple) = uniform_rand(stateset[1], stateset[2])
@inline sample_from_stateset(::Type{Discrete}, stateset::Tuple, num) = rand(stateset, num)
@inline sample_from_stateset(::Type{Continuous}, stateset::Tuple, num) = uniform_rand(stateset[1], stateset[2], num)


@inline sample_from_stateset(::Any, rng::AbstractRNG, stateset::AbstractVector) = rand(rng, stateset)
@inline sample_from_stateset(::Type{Discrete}, rng::AbstractRNG, stateset::Tuple) = rand(rng, stateset)
@inline sample_from_stateset(::Type{Continuous}, rng::AbstractRNG, stateset::Tuple) = uniform_rand(rng, stateset[1], stateset[2])

@inline sampleState(::Any, oldstate, rng, stateset) = rand(stateset)
@inline sampleState(::Type{Discrete}, oldstate, rng, stateset) = oldstate == stateset[1] ? stateset[2] : stateset[1]
@inline sampleState(::Type{Continuous}, oldstate, rng, stateset) = sample_from_stateset(Continuous, rng, stateset)

## Iterator that changes without having to recompile the loop
struct DynamicalIterator{GT}
    g::GT
end

function Base.rand(rng::MersenneTwister, it::DynamicalIterator)
    return rand(rng, it.g.defects.aliveList)
end