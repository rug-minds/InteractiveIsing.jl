export LayeredMetropolis

struct LayeredMetropolis <: MCAlgorithm end
requires(::Type{LayeredMetropolis}) = Δi_H()


function reserved_symbols(::Type{LayeredMetropolis})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

function _prepareOLD(::Type{LayeredMetropolis}, g; kwargs...)
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)
    prepared_kwargs = pairs((;g,
                        gstate = g.state,
                        gadj = g.adj,
                        gparams = g.params,
                        iterator = ising_it(g, g.stype),
                        layers = unshuffled(g.layers),
                        ΔH
                    ))
    return prepared_kwargs
end

function _prepare(::Type{LayeredMetropolis}, @specialize(args))
    (;g) = args
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)
    prepared_args = pairs((;g,
                        gstate = g.state,
                        gadj = g.adj,
                        gparams = g.params,
                        iterator = ising_it(g, g.stype),
                        layers = unshuffled(g.layers),
                        ΔH
                    ))
    return prepared_args
end


Base.@propagate_inbounds @inline function LayeredMetropolis(@specialize(args))
    #Define vars
    (;layers, iterator) = args
    i = rand(rng, iterator)
    @inline layerswitch(LayeredMetropolis, i, layers, args)
end

@inline function LayeredMetropolis(i, args, layertype)
    (;g, gstate, gadj, gparams, ΔH) = args
    @inline Metropolis(i, g, gstate, gadj, gparams, ΔH, layertype)
end
