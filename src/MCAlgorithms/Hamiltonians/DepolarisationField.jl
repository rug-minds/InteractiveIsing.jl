struct DepolField{PV <: ParamVal} <: Hamiltonian 
    dpf::PV
    left_layers::Int32
    right_layers::Int32
end

DepolField(g, left_layers = 1, right_layers = 1) = DepolField(ParamVal(eltype(g),eltype(g) |> zero, "Depolarisation Field", true), left_layers, right_layers)

function update!(dpf::ParamVal{DepolField}, args)
    (;g, j, Δs_i) = args
    layer2dsize = size(g,1)*size(g,2)
    if j <= layer2dsize*dpf.left_layers || j > layer2dsize*(size(g,3)-dpf.right_layers)
        dpf.dpf[j] = dpf.dpf[j] + Δs_i
    end
end

@ParameterRefs function deltaH(::DepolField)
    return :dpf_j*(s_j-sn_j)
end

