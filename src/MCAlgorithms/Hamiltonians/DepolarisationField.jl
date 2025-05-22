export DepolField
struct DepolField{PV <: ParamVal, CV <: ParamVal} <: Hamiltonian 
    dpf::PV
    c::CV
    left_layers::Int32
    right_layers::Int32
end



function get_dpf(dpf, g)
    ll = dpf.left_layers
    rl = dpf.right_layers
    if length(size(g[1])) == 2
        return sum(state(g[1])[:,1:ll])
    else
        layer_2dsize = size(g[1],1)*size(g[1],2)
        return sum(state(g)[1:layer_2dsize*ll]) + sum(state(g)[end+1 - layer_2dsize*rl: end])
    end
end

function init!(dpf::DepolField, g)
    dpf.dpf[] = get_dpf(dpf, g)
    return dpf
end

function DepolField(g; c = 3600, left_layers = 1, right_layers = 1)
    pv = GlobalParamVal(eltype(g) |> zero, length(state(g[1])), "Depolarisation Field", true)
    cv = DefaultParamVal(eltype(g)(c), "Depolarisation Field")
    println(pv.runtimeglobal)
    dpf = DepolField(pv, cv, Int32(left_layers), Int32(right_layers))
    init!(dpf, g)
    return dpf
end

function update!(dpf::DepolField, args)
    (;lmeta, j, Δs_i) = args
    l1 = layer(lmeta)
    layer2dsize = size(l1,1)*size(l1,2)
    if j <= layer2dsize*dpf.left_layers || j > layer2dsize*(size(l1,3)-dpf.right_layers)
        dpf.dpf[j] = dpf.dpf[j] + Δs_i[]
    end
    return 
end

@ParameterRefs function deltaH(::DepolField)
    return (dpf_j/c_)*(s_j-sn_j)
end

