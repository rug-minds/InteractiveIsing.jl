export Metropolis

struct Metropolis <: MCAlgorithm end
requires(::Type{Metropolis}) = Δi_H


function reserved_symbols(::Type{Metropolis})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end


function _prepare(::Type{Metropolis}, g; kwargs...)
    prepared_kwargs = pairs((;g,
                        gstate = g.state,
                        gadj = g.adj,
                        gparams = g.params,
                        iterator = ising_it(g, g.stype),
                        rng = MersenneTwister(),
                        gstype = g.stype,
                        layertype = typeof(g.layers[1]),
                        H = Hamiltonian_Builder(Metropolis, g, g.hamiltonians...),
                    ))
    return prepared_kwargs
end

Base.@propagate_inbounds @inline function Metropolis(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, H, layertype) = args
    i = rand(rng, iterator)
    Metropolis(i, g, gstate, gadj, gparams, H, layertype)
end

@inline function Metropolis(i, g, @specialize(gstate::Vector{T}), gadj, gparams, @specialize(ΔH), @specialize(layertype::Type{IsingLayer{StateType, StateSet, C, D}})) where {T, StateType, StateSet, C, D}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[i]
    
    newstate = sampleState(StateType, oldstate, rng, StateSet)   

    ΔE = @inline ΔH(i, gstate, newstate, oldstate, gadj, gparams, layertype)
    efac = exp(-β*ΔE)
    randnum = rand(rng, Float32)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[i] = newstate 
    end
    
    return nothing
end



# @inline function updateMetropolis(g, gstate::Vector{T}, gadj, gparams, ΔH)
#     β = 1f0/(temp(g))

#     oldstate = @inbounds gstate[idx]
    
#     ΔE = ΔH(g, oldstate, 1, gstate, gadj, idx, gstype, Discrete)

#     if (ΔE <= 0f0 || rand(rng, Float32) < exp(-β*ΔE))
#         @inbounds gstate[idx] *= -Int8(1)
#     end
#     return nothing
# end





