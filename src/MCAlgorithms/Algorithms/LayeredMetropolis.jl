export LayeredMetropolis

struct LayeredMetropolis <: MCAlgorithm end
# requires(::Type{LayeredMetropolis}) = Δi_H()
requires(::LayeredMetropolis) = deltH()


function reserved_symbols(::Type{LayeredMetropolis})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

function Processes.prepare(::LayeredMetropolis, @specialize(args))
    (;g) = args
    ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian)

    return (; gstate = g.state,
            gadj = g.adj,
            gparams = g.params,
            iterator = ising_it(g),
            layers = unshuffled(g.layers),
            ΔH,
            rng = MersenneTwister(),
            layerarch = GetArchitecture(unshuffled(g.layers)...),
            Ms = [Ref(sum(state(g[i]))) for i in 1:length(g.layers)]
        )
end


Base.@propagate_inbounds @inline function (::LayeredMetropolis)(@specialize(args))
    #Define varso
    
    (;iterator, rng, layerarch) = args
    i = rand(rng, iterator)
    @inline layerswitch(LayeredMetropolis, i, layerarch, args)
end

@inline function LayeredMetropolis(i, args, layeridx, lmeta)
    (;g, gstate, gadj, gparams, ΔH, rng, Ms) = args
    @inline Metropolis(args, i, g, gstate, gadj, gparams, Ms[layeridx], ΔH, rng, lmeta)
end