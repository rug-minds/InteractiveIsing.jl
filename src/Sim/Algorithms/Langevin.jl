function updateLangevin end
function updateLangevinThreaded end
export updateLangevin

using Distributions: Normal
const stepsize = Ref(0.01f0)
# PREALLOCATED MEMORY
const langevin_prealloc  = Float32[]
setstepsize(val) = stepsize[] = val
export setstepsize


function prepare(::Union{typeof(updateLangevin), typeof(updateLangevinThreaded)}, g; kwargs...)
    resize!(langevin_prealloc, length(state(g)))
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = sp_adj(g),
                        gparams = params(g),
                        iterator = ising_it(g, stype(g)),
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
    noise = rand(Normal(0f0,1f0), length(iterator))
    @inbounds (@view gstate[iterator]) .= clamp!((@view gstate[iterator]) - (stepsize[])*grad + sqrt(2f0*stepsize[]*temp(g))*noise, -1f0, 1f0)
end

function updateLangevin(g::IsingGraph, gstate, gadj, iterator, rng, gstype::ST, dEFunc) where {ST <: SType}
    for (i_idx, s_idx) in enumerate(iterator)
        langevin_prealloc[i_idx] = dEFunc(g, gstate, gadj, s_idx, gstype)
    end
    grad = @view langevin_prealloc[1:length(iterator)]
    noise = rand(Normal(0f0,1f0), length(iterator))
    @inbounds (@view gstate[iterator]) .= clamp!((@view gstate[iterator]) - (stepsize[])*grad + sqrt(2f0*stepsize[]*temp(g))*noise, -1f0, 1f0)
end

