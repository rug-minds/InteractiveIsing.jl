export LayeredMetropolis

struct LayeredMetropolis <: MCAlgorithm end
requires(::Type{LayeredMetropolis}) = Δi_H()


function reserved_symbols(::Type{LayeredMetropolis})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

function Processes.prepare(::LayeredMetropolis, @specialize(args))
    (;g) = args
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)

    return (;g,
            gstate = g.state,
            gadj = g.adj,
            gparams = g.params,
            iterator = ising_it(g, g.stype),
            layers = unshuffled(g.layers),
            ΔH,
            rng = MersenneTwister()
        )
end


Base.@propagate_inbounds @inline function LayeredMetropolis(@specialize(args))
    #Define vars
    (;layers, iterator, rng) = args
    i = rand(rng, iterator)
    @inline layerswitch(LayeredMetropolis, i, layers, args)
end

@inline function LayeredMetropolis(i, args, layertype)
    (;g, gstate, gadj, gparams, ΔH) = args
    @inline Metropolis(i, g, gstate, gadj, gparams, ΔH, layertype)
end
