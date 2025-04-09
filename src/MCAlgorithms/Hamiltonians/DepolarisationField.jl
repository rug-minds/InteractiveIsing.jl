export DepolField
struct DepolField{PV <: ParamVal} <: Hamiltonian 
    dpf::PV
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

function DepolField(g; left_layers = 1, right_layers = 1)
    pv = GlobalParamVal(eltype(g) |> zero, "Depolarisation Field", true)
    println(pv.runtimeglobal)
    dpf = DepolField(pv, Int32(left_layers), Int32(right_layers))
    init!(dpf, g)
    return dpf
end

function update!(dpf::DepolField, args)
    (;g, j, Δs_i) = args
    layer2dsize = size(g[1],1)*size(g[1],2)
    if j <= layer2dsize*dpf.left_layers || j > layer2dsize*(size(g[1],3)-dpf.right_layers)
        dpf.dpf[j] = dpf.dpf[j] + Δs_i
    end
end

@ParameterRefs function deltaH(::DepolField)
    return dpf_j/1800*(s_j-sn_j)
end

