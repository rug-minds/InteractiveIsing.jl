
"""
Fallback preparation for updateFunc
"""
prepare(::Any, ::Any) = nothing

@inline function updateMetropolis(@specialize(args))
    #Define vars
    (;g, gstate, gadj, iterator, rng, gstype, ΔEFunc) = args
    idx = rand(rng, iterator)
    updateMetropolis(idx, g, gstate, gadj, rng, gstype, ΔEFunc)
end


@inline function updateMetropolis(idx::Integer, g, gstate::Vector{Int8}, gadj, rng, gstype::ST, ΔEFunc) where {ST <: SType}

    beta = 1f0/(temp(g))

    oldstate = @inbounds gstate[idx]
    
    ΔE = ΔEFunc(g, oldstate, 1, gstate, gadj, idx, gstype)

    if (ΔE <= 0f0 || rand(rng, Float32) < exp(beta*minEdiff))
        @inbounds gstate[idx] *= -Int8(1)
    end
    return nothing
end


@inline function updateMetropolis(idx::Integer, g, gstate::Vector{Float32}, gadj, rng, gstype::ST, ΔEFunc) where {ST <: SType}

    beta = 1f0/(temp(g))

    oldstate = @inbounds gstate[idx]

    newstate = 2f0*(rand(rng, Float32)- .5f0)

    ΔE = ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, gstype)

    if (ΔE < 0f0 || rand(rng, Float32) < exp(-beta*ΔE))
        @inbounds g.state[idx] = newstate 
    end

    return nothing
end

function get_args(::typeof(updateMetropolis))
    return (:g, :gstate, :gadj, :iterator, :rng, :gstype, :ΔEFunc)
end

@inline function ΔEIsing(g::IsingGraph{Int8}, oldstate, newstate, gstate, gadj, idx, @specialize(gstype))
    return @inbounds 2f0*oldstate*dEIsing(g, gstate, gadj, idx, gstype)
end
@inline function ΔEIsing(g::IsingGraph{Float32}, oldstate, newstate, gstate, gadj, idx, @specialize(gstype))

    efactor = dEIsing(g, gstate, gadj, idx, gstype)
    return EdiffIsing(g, gstype, idx, efactor, oldstate, newstate)
end

function prepare(::typeof(updateMetropolis), g; kwargs..., )::NamedTuple
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = sp_adj(g),
                        iterator = ising_it(g, stype(g)),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        ΔEFunc = ΔEIsing,
                    ))
    return (;replacekwargs(def_kwargs, kwargs)...)
end

export updateMetropolis



using Distributions: Normal
const stepsize = Ref(0.01f0)
setstepsize(val) = stepsize[] = val
export setstepsize

const langevin_prealloc  = Float32[]
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
function prepare(::Union{typeof(updateLangevin), typeof(updateLangevinThreaded)}, g)
    resize!(langevin_prealloc, length(state(g)))
end
export updateLangevin

let times = Ref([])
    global function upDebug(g, params, lTemp, gstate::Vector, gadj, iterator, rng, gstype, dEFunc)

        beta = 1/(lTemp[])
        
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

        if (Estate >= 0 || rand(rng) < exp(beta*minEdiff))
            @inbounds g.state[idx] *= -1
        end
        
    end
end
export upDebug

