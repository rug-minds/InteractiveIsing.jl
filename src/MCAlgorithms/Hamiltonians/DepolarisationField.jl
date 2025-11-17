export DepolField
struct DepolField{PV <: ParamVal, CV <: ParamVal, SP, T} <: Hamiltonian 
    dpf::PV
    c::CV
    field_adj::SP
    top_layers::Int32
    bottom_layers::Int32
    size::T
end

function Base.in(j::Integer, dpf::DepolField)
    j <= prod(dpf.size[1:end-1])*dpf.top_layers || j > prod(dpf.size[1:end-1])*(dpf.size[end]-dpf.bottom_layers)
end


function get_dpf(dpf, g)
    ll = dpf.top_layers
    rl = dpf.bottom_layers
    if length(size(g[1])) == 2
        return -sum(state(g[1])[:,1:ll])
    else
        layer_2dsize = size(g[1],1)*size(g[1],2)
        return -sum(state(g)[1:layer_2dsize*ll]) + sum(state(g)[end+1 - layer_2dsize*rl: end])
    end
end

function init!(dpf::DepolField, g)
    dpf.dpf[] = get_dpf(dpf, g)
    return dpf
end

function DepolField(g; top_layers = 1, bottom_layers = 1, c = prod(size(g[1])[1:end-1])*(top_layers+bottom_layers) )
    pv = HomogenousParamVal(eltype(g) |> zero, length(state(g[1])), true, description = "Depolarisation Field")
    # cv = DefaultParamVal(eltype(g)(c), description = "Depolarisation Field")
    cv = ScalarParam(eltype(g), c; description = "Depolarisation Field")
    wg = @WG (dr) -> 1/dr^3 NN = 3
    fv = sparse(genLayerConnections(g[1], wg)..., nstates(g[1]), nstates(g[1]))
    
    dpf = DepolField(pv, cv, fv, Int32(top_layers), Int32(bottom_layers), size(g[1]))
    init!(dpf, g)
    return dpf
end

function update!(::Metropolis, dpf::DepolField, args)
    (;lmeta, j, Δs_j) = args
    l1 = layer(lmeta)
    layer2dsize = size(l1,1)*size(l1,2)
    if j <= layer2dsize*dpf.top_layers || j > layer2dsize*(size(l1,3)-dpf.bottom_layers)
        dpf.dpf[j] = dpf.dpf[j] - Δs_j[]
    end
    return 
end

# ΔH_expr[DepolField] = :( (dpf[j]/c[])*(s[j]) )

function ΔH(dpf::DepolField, args, drule)
    (;s, self, c) = args
    j = getidx(drule)
    if j ∈ dpf
        field_delta = zero(eltype(s))
        @turbo for ptr in nzrange(dpf.field_adj, j)
            i = dpf.field_adj.rowval[ptr]
            w_ij = dpf.field_adj.nzval[ptr]
            field_delta += w_ij * s[i]
        end
        # s is in surface
        # So dpf at all I changes. Dpf is proportional to - s_j, and the distances are stored in field_adj
        # H_j = - c (F_j*s_j - ∑_i s_i fw_ij s_j)
        # ΔE =  - c (F_j*(s_j' - s_j) - ∑_i s_i fw_ij (s_j' - s_j)) 
        # ΔE = c *(F_j - ∑_i s_i fw_ij) * (s_j - s_j')

        return (dpf.dpf[j]/c[] - field_delta)*(s[j] - drule[]) 
    else

        # H_j = - c* F_j*s_j
        # ΔE = - c * F_j * (s_j' - s_j) = c * F_j * (s_j - s_j')
        return (dpf.dpf[j]/c[])*(s[j] - drule[])

    end
end

@ParameterRefs function deltaH(::DepolField)
    return (dpf[j]/c[])*(sn[j]-s[j])
end

