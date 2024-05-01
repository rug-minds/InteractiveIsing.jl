export LayeredMetropolis

struct LayeredMetropolis <: MCAlgorithm end
requires(::LayeredMetropolis) = Δi_H


function reserved_symbols(::Type{LayeredMetropolis})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

function _prepare(::Type{LayeredMetropolis}, g; kwargs...)
    prepared_kwargs = pairs((;g,
                        gstate = g.state,
                        gadj = g.adj,
                        gparams = g.params,
                        iterator = ising_it(g, g.stype),
                        layers = unshuffled(g.layers),
                        H = Hamiltonian_Builder(Metropolis, g, g.hamiltonians...),
                    ))
    return prepared_kwargs
end

Base.@propagate_inbounds @inline function LayeredMetropolis(@specialize(args))
    #Define vars
    (;layers, iterator) = args
    i = rand(rng, iterator)
    @inline layerswitch(LayeredMetropolis, i, layers, args)
end

@inline function LayeredMetropolis(i, @specialize(args), @specialize(layertype))
    (;g, gstate, gadj, gparams, H) = args
    @inline Metropolis(i, g, gstate, gadj, gparams, H, layertype)
end
