function layeredLangevin end
const llangevin_prealloc  = Float32[]

function get_args(::typeof(layeredLangevin))
    return (:g, :gstate, :gadj, :gparams, :iterator, :rng, :gstype, :ΔEFunc)
end

function prepare(::typeof(layeredLangevin), g; kwargs...)
    resize!(llangevin_prealloc, length(state(g)))
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = sp_adj(g),
                        gparams = params(g),
                        params = g.params,
                        giterator = ising_it(g, stype(g)),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        ΔEFunc = ΔEIsing,
                        layers = layers(g).data,
                        l_iterators = l_iterators,
                    ))

    # Extremise discrete states
    extremiseDiscrete!.(layers(g))
    return (;replacekwargs(def_kwargs, kwargs)...)
end

@inline @generated function layeredLangevin(@specialize(args))
    expr = Meta.parse(gen_exp(layeredLangevin, args))
    return expr
end


function gen_exp(::typeof(layeredLangevin), argstype)
    expr_str = "begin 
    $(get_args_string(layeredMetropolis)) = args
        $(generate_layerswitch(layeredMetropolis, grouped_ltypes_idxs(argstype)))
    end"
end

## TODO: Not done
# function generate_layerswitch(::typeof(layeredLangevin), typeidx_zip)
#     statements += 1
#     expr_str = "lastidx = 1
#     for stateidx in iterator
#         "
#     for (layeridx, ltypeidx) in typeidx_zip
#         expr_str *=if (idx <= lastidx + $(layeridx))
#             $(generate_layerupdate(layeredMetropolis, ltypeidx))
#         end
#     end
#     return expr_str
# end