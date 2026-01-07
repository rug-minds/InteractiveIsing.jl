export LayeredMetropolis

# struct LayeredMetropolis <: MCAlgorithm end
# requires(::Type{LayeredMetropolis}) = Δi_H()
# requires(::LayeredMetropolis) = deltH()


# function reserved_symbols(::Type{LayeredMetropolis})
#     return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
# end

@ProcessAlgorithm function LayeredMetropolis(iterator, rng, layers, args)
    #Define varso
    
    # (;iterator, rng, layerarch) = args
    j = rand(rng, iterator)
    @inline layerswitch(Layered_step, j, layers, args)
end

function Processes.prepare(::LayeredMetropolis, args::A) where A
    (;g) = args
    args = prepare(Metropolis(), args)
    Ms = [Ref(sum(state(g[i]))) for i in 1:length(g.layers)]

    return (;args..., 
            layers = g.layers,
            Ms = Ms)
end

Base.@propagate_inbounds @inline function Layered_step(j, args::NamedTuple, layeridx, layer)
    M = args.Ms[layeridx]
    args = (;args..., j, layeridx, M, layer)
    @inline Metropolis_step(args)
end