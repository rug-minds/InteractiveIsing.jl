function updateLangevin end
function updateLangevinThreaded end
export updateLangevin

using Distributions: Normal
const stepsize = Ref(0.01f0)
# PREALLOCATED MEMORY
const langevin_prealloc64  = Float64[]
setstepsize(val) = stepsize[] = val
export setstepsize


function Processes.init(::Union{typeof(updateLangevin), typeof(updateLangevinThreaded)}, g; kwargs...)
    resize!(langevin_prealloc, length(state(g)))
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = adj(g),
                        gparams = params(g),
                        iterator = stateiterator(g),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        dEFunc = dEIsing,
                    ))
    return (;replacekwargs(def_kwargs, kwargs)...)
end

@inline function updateLangevin(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, rng, gstype, dEFunc) = args
    updateLangevin(g, gstate, gadj, iterator, rng, gstype, dEFunc)
end


function updateLangevinThreaded(g::IsingGraph, gstate, gadj, iterator, rng, gstype::ST, dEFunc) where {ST <: SType}
    Threads.@threads for (i_idx, s_idx) in collect(enumerate(iterator))
        langevin_prealloc[i_idx] = dEFunc(g, gstate, gadj, s_idx, gstype)
    end
    grad = @view langevin_prealloc[1:length(iterator)]
    noise = rand(Normal(zero(T),one(T)), length(iterator))
    @inbounds (@view gstate[iterator]) .= clamp!((@view gstate[iterator]) - (stepsize[])*grad + sqrt(T(2)*stepsize[]*temp(g))*noise, -one(T), one(T))
end

function updateLangevin(g::IsingGraph{T}, gstate, gadj, iterator, rng, gstype::ST, dEFunc) where {T,ST <: SType}
    for (i_idx, s_idx) in enumerate(iterator)
        langevin_prealloc[i_idx] = dEFunc(g, gstate, gadj, s_idx, gstype)
    end
    grad = @view langevin_prealloc[1:length(iterator)]
    noise = rand(Normal(zero(T),one(T)), length(iterator))
    @inbounds (@view gstate[iterator]) .= clamp!((@view gstate[iterator]) - (stepsize[])*grad + sqrt(T(2)*stepsize[]*temp(g))*noise, -one(T), one(T))
end

