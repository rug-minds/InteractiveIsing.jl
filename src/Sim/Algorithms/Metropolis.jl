function updateMetropolis end

function Processes.prepare(::typeof(updateMetropolis), g; kwargs...)
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = adj(g),
                        gparams = params(g),
                        iterator = ising_it(g),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        ΔEFunc = ΔEIsing,
                    ))
    return (;replacekwargs(def_kwargs, kwargs)...)
end

@inline function updateMetropolis(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, rng, gstype, ΔEFunc) = args
    idx = rand(rng, iterator)
    updateMetropolis(idx, g, gstate, gadj, rng, gstype, ΔEFunc)
end


export updateMetropolis


@inline function updateMetropolis(idx::Integer, g, gstate::Vector{Int8}, gadj, rng, gstype::ST, ΔEFunc) where {ST <: SType}

    β = 1f0/(temp(g))

    oldstate = @inbounds gstate[idx]
    
    ΔE = @inline ΔEFunc(g, oldstate, 1, gstate, gadj, idx, gstype, Discrete)

    if (ΔE <= 0f0 || rand(rng, Float32) < exp(-β*ΔE))
        @inbounds gstate[idx] *= -Int8(1)
    end
    return nothing
end


@inline function updateMetropolis(idx::Integer, g, gstate::Vector{T}, gadj, rng, gstype::ST, ΔEFunc) where {ST <: SType, T <: AbstractFloat}

    β = one(T)/(temp(g))

    oldstate = @inbounds gstate[idx]

    newstate = T(2)*(rand(rng, Float32)- T(0.5))

    ΔE = @inline ΔEFunc(g, oldstate, newstate, gstate, gadj, idx, gstype, Continuous)
    if (ΔE < zero(T) || rand(rng, Float32) < exp(-β*ΔE))
        @inbounds g.state[idx] = newstate 
    end

    return nothing
end

function get_args(::typeof(updateMetropolis))
    return (:g, :gstate, :gadj, :iterator, :rng, :gstype, :ΔEFunc)
end