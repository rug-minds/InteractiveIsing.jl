export DepolField
struct DepolField{PV <: ParamVal, CV <: ParamVal} <: Hamiltonian 
    dpf::PV
    c::CV
    top_layers::Int32
    bottom_layers::Int32
end



function get_dpf(dpf, g)
    ll = dpf.top_layers
    rl = dpf.bottom_layers
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

function DepolField(g; c = 3600, top_layers = 1, bottom_layers = 1)
    pv = HomogenousParamVal(eltype(g) |> zero, length(state(g[1])), true, description = "Depolarisation Field")
    cv = DefaultParamVal(eltype(g)(c), description = "Depolarisation Field")
    println(pv.homogenousval)
    dpf = DepolField(pv, cv, Int32(top_layers), Int32(bottom_layers))
    init!(dpf, g)
    return dpf
end

function update!(dpf::DepolField, args)
    (;lmeta, j, Δs_j) = args
    l1 = layer(lmeta)
    layer2dsize = size(l1,1)*size(l1,2)
    if j <= layer2dsize*dpf.top_layers || j > layer2dsize*(size(l1,3)-dpf.bottom_layers)
        dpf.dpf[j] = dpf.dpf[j] + Δs_j[]
    end
    return 
end

ΔH_expr[DepolField] = :( (dpf[j]/c[])*(s[j]) )

@ParameterRefs function deltaH(::DepolField)
    return (dpf[j]/c[])*(sn[j]-s[j])
end

