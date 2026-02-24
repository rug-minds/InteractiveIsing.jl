export DepolField
struct DepolField{PV <: ParamTensor, CV <: ParamTensor, F, T} <: HamiltonianTerm 
    dpf::PV
    c::CV
    zfunc::F
    # field_adj::SP
    top_layers::Int32
    bottom_layers::Int32
    size::T
end

function DepolField(g; top_layers = 1, bottom_layers = top_layers, c = 1/prod(size(g[1])[1:end-1])*(top_layers+bottom_layers), zfunc = z -> exp(-(abs(min(abs(z-1), abs(z-size(g[1])))))))
    pv = HomogeneousParam(eltype(g)(0), length(state(g[1])), description = "Depolarisation Field")
    cv = ScalarParam(eltype(g), c; description = "Depolarisation Field")
    # wg = @WG (dr) -> 1/dr^3 NN = NN
    # fv = sparse(genLayerConnections(g[1], wg)..., nstates(g[1]), nstates(g[1]))
    
    dpf = DepolField(pv, cv, zfunc, Int32(top_layers), Int32(bottom_layers), size(g[1]))
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

function update!(::Metropolis, dpf::DepolField, params)
    (;j, proposal) = params
    if isaccepted(proposal)
        if j ∈ dpf
            z = layers_deep(j, dpf)
            dpf.dpf[j] -= dpf.zfunc(z)*delta(proposal)
        end
    end
    return 
end

# ΔH_expr[DepolField] = :( (dpf[j]/c[])*(s[j]) )

# function ΔH(dpf::DepolField, params, proposal)
function calculate(::ΔH, dpf::DepolField, hargs, proposal)
    (;s, self, c) = params
    j = getidx(proposal)
    base_term = 1/2*c[]*dpf.dpf[j]
    if j ∈ dpf # If in the surface
                # Also compute the effect of changhing the field
        # field_delta = zero(eltype(s))
        # @turbo for ptr in nzrange(dpf.field_adj, j)
        #     i = dpf.field_adj.rowval[ptr]
        #     w_ij = dpf.field_adj.nzval[ptr]
        #     field_delta += w_ij * s[i]
        # end
        # z = layers_deep(j, dpf)
        # base_term -= dpf.zfunc(z)*field_delta
        base_term *= 2
    end

    return base_term * (s[j] - proposal[])
end

# function ΔH(dpf::DepolField, g::PottsGraph, args, proposal)
    
# end

# @ParameterRefs function deltaH(::DepolField)
#     return (dpf[j]/c[])*(sn[j]-s[j])
# end

function Base.show(io::IO, ::MIME"text/plain", dpf::DepolField)
    println(io, "DepolField Hamiltonian")
    println(io, "  top_layers: $(dpf.top_layers)")
    println(io, "  bottom_layers: $(dpf.bottom_layers)")
    println(io, "  size: $(dpf.size)")
    println(io, "  field_adj: $(size(dpf.field_adj)) sparse matrix with $(nnz(dpf.field_adj)) nonzeros")
    println(io, "  dpf: ", dpf.dpf)
    println(io, "  c: ", dpf.c)
    print(io, "  field_c: ", dpf.field_c)
end