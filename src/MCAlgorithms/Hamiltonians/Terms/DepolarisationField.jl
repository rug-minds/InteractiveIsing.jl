export DepolField
struct DepolField{PV <: ParamTensor, CV <: ParamTensor, F, T, R} <: HamiltonianTerm 
    dpf::PV
    c::CV
    zfunc::F
    # field_adj::SP
    top_layers::Int32
    bottom_layers::Int32
    size::T
    M::Base.RefValue{R}
    surface_NxNy::Int
end

Base.Expr(::DepolField) = :( -(dpf[j]/(surface_NxNy * c[]))*(s[j]) )

function DepolField(g; top_layers = 1, bottom_layers = top_layers, c = 1f0, zfunc = z -> exp(-(min(abs(z-1), abs(z-size(g[1],3))))))
    pv = HomogeneousParam(eltype(g)(0), length(state(g[1])), description = "Depolarisation Field")
    cv = ScalarParam(eltype(g), c; description = "Depolarisation Field")
    # wg = @WG (dr) -> 1/dr^3 NN = NN
    # fv = sparse(genLayerConnections(g[1], wg)..., nstates(g[1]), nstates(g[1]))
    NxNy = prod(size(g[1])[1:end-1])
    surface_NxNy = (bottom_layers+top_layers)*NxNy
    dpf = DepolField(pv, cv, zfunc, Int32(top_layers), Int32(bottom_layers), size(g[1]), Ref(eltype(g)(sum(state(g[1])))), surface_NxNy)
    init!(dpf, g)
    return dpf
end

Base.CartesianIndices(df::DepolField) = CartesianIndices(df.size)
Base.LinearIndices(df::DepolField) = LinearIndices(df.size)

get_top_layers(dpf::DepolField, g) = @view state(g[1])[CartesianIndices(dpf.size)[:,:,1:dpf.top_layers]]
get_bottom_layers(dpf::DepolField, g) = @view state(g[1])[CartesianIndices(dpf.size)[:,:,end - dpf.bottom_layers +1:end]]
 

function Base.in(j::Integer, dpf::DepolField)
    CI = CartesianIndices(dpf.size)
    CI[j] ∈ CI[:,:,1:dpf.top_layers] || CI[j] ∈ CI[:,:,end - dpf.bottom_layers +1:end]
end

"""
How many layers from the top or bottom is index j in dpf
"""
function layers_deep(j, dpf::DepolField)
    z = CartesianIndices(dpf.size)[j].I[end]
    z = z >= round(Int,dpf.size[end]/2) ? (dpf.size[end] - z + 1) : z
    return z
end

"""
Get the total Depolarisation (sum of all boundary layer spins scaled by zfunc)
"""
function get_dpf(dpf, g)
    ll = dpf.top_layers
    rl = dpf.bottom_layers
    if length(size(g[1])) == 2
        return 
    else # 3Dim

        # Count every 2D layer, scale by zfunc
        CI = CartesianIndices(dpf)
        top_layers = get_top_layers(dpf, g)
        bottom_layers = get_bottom_layers(dpf, g)
        total = zero(eltype(g))
        for slice in vcat(eachslice(top_layers, dims = 3), eachslice(bottom_layers, dims = 3))
            z = slice.indices[end]
            z = z >= round(Int,dpf.size[end]/2) ? (dpf.size[end] - z + 1) : z
            total += dpf.zfunc(z) * sum(slice)
        end

        return -total
    end
end

function init!(dpf::DepolField, g)
    dpf.dpf[] = get_dpf(dpf, g)
    return dpf
end

function update!(::Metropolis, dpf::DepolField, context)
    (;proposal) = context
    if isaccepted(proposal)
        j = at_idx(proposal)
        if j ∈ dpf
            z = layers_deep(j, dpf)
            dpf.dpf[] -= dpf.zfunc(z) * delta(proposal)
        end
        dpf.M[] += delta(proposal)
    end
    return 
end

# ΔH_expr[DepolField] = 

# function ΔH(dpf::DepolField, params, proposal)
function calculate(::ΔH, dpf::DepolField, hargs, proposal)
    j = at_idx(proposal)
    T = eltype(hargs.s)
    ΔM = delta(proposal)
    D = dpf.dpf[]/dpf.surface_NxNy
    M = dpf.M[]
    ΔD = zero(T)
    if j ∈ dpf # In surface
        z = layers_deep(j, dpf)
        ΔD = -T(dpf.zfunc(z)) * ΔM / dpf.surface_NxNy
    end
    c = hargs.c[]
    return -(D * ΔM + M * ΔD + ΔD * ΔM) / c
end

# function calculate(::ΔH, dpf::DepolField, hargs, proposal)
#     j = at_idx(proposal)
#     if !(j ∈ dpf)
#         return zero(eltype(hargs.s))
#     end

#     T = eltype(hargs.s)
#     z = layers_deep(j, dpf)
#     ΔD = -T(dpf.zfunc(z)) * delta(proposal)
#     D = dpf.dpf[]
#     c = hargs.c[]
#     return (D * ΔD + T(0.5) * (ΔD^2)) / c
# end


# function calculate(::ΔH, dpf::DepolField, hargs, proposal)
#     (;s, self, c) = hargs
#     j = getidx(proposal)
#     base_term = 1/2*c[]*dpf.dpf[j]
#     if j ∈ dpf # If in the surface
#                 # Also compute the effect of changhing the field
#         # field_delta = zero(eltype(s))
#         # @turbo for ptr in nzrange(dpf.field_adj, j)
#         #     i = dpf.field_adj.rowval[ptr]
#         #     w_ij = dpf.field_adj.nzval[ptr]
#         #     field_delta += w_ij * s[i]
#         # end
#         # z = layers_deep(j, dpf)
#         # base_term -= dpf.zfunc(z)*field_delta
#         base_term *= 2
#     end

#     return base_term * (s[j] - proposal[])
# end
