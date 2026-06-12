export DepolField
struct DepolInternal{PV,F,S,R} <: InternalImplementation
    dpf::PV
    zfunc::F
    top_layers::Int32
    bottom_layers::Int32
    size::S
    M::Base.RefValue{R}
    surface_NxNy::Int
end

struct DepolField{P,I} <: LayerTerm
    layer::Int
    parameters::P
    internal::I
end

Base.Expr(::DepolField) = :( -(dpf[j]/(surface_NxNy * c[]))*(s[j]) )

@inline function DepolField(; layer = 1, top_layers = 1, bottom_layers = top_layers, c = 1f0, zfunc = z -> exp(-abs(z - 1)))
    params = Parameters(
        parameter(;
            c,
            type = AbstractArray,
            default = ConstVal(1f0),
            ensure = ensure_isinggraph_scalar,
            info = "Depolarisation coupling constant",
            units = physicalunits(energy = 1, role = :depolarisation_energy),
        ),
    )
    internal = InternalPlan((; top_layers, bottom_layers, zfunc)) do plan, g
        config = plan.values
        T = eltype(g)
        pv = HomogeneousParam(zero(T), length(state(g)); description = "Depolarisation Field")
        layer_size = size(g)
        nxny = prod(layer_size[1:end-1])
        surface_NxNy = (config.bottom_layers + config.top_layers) * nxny

        return DepolInternal(
            pv,
            config.zfunc,
            Int32(config.top_layers),
            Int32(config.bottom_layers),
            layer_size,
            Ref(T(sum(state(g)))),
            surface_NxNy,
        )
    end
    return DepolField(Int(layer), params, internal)
end

@inline DepolField(layer::Integer; kwargs...) = DepolField(; layer, kwargs...)

@inline function DepolField(g; layer = 1, top_layers = 1, bottom_layers = top_layers, c = 1f0, zfunc = z -> exp(-(min(abs(z-1), abs(z-size(g[layer],3))))))
    h = instantiate(
        DepolField(; layer, top_layers, bottom_layers, c, zfunc),
        g,
    )
    return init!(h, g)
end

@inline Base.CartesianIndices(df::DepolField) = CartesianIndices(df.size)
@inline Base.LinearIndices(df::DepolField) = LinearIndices(df.size)

@inline get_top_layers(dpf::DepolField, layer) = @view state(layer)[CartesianIndices(dpf.size)[:,:,1:dpf.top_layers]]
@inline get_bottom_layers(dpf::DepolField, layer) = @view state(layer)[CartesianIndices(dpf.size)[:,:,end - dpf.bottom_layers +1:end]]
 

@inline function Base.in(j::Integer, dpf::DepolField)
    CI = CartesianIndices(dpf.size)
    CI[j] ∈ CI[:,:,1:dpf.top_layers] || CI[j] ∈ CI[:,:,end - dpf.bottom_layers +1:end]
end

"""
How many layers from the top or bottom is index j in dpf
"""
@inline function layers_deep(j, dpf::DepolField)
    z = CartesianIndices(dpf.size)[j].I[end]
    z = z >= round(Int,dpf.size[end]/2) ? (dpf.size[end] - z + 1) : z
    return z
end

"""
Get the total Depolarisation (sum of all boundary layer spins scaled by zfunc)
"""
@inline function get_dpf(dpf, g)
    ll = dpf.top_layers
    rl = dpf.bottom_layers
    if length(size(g)) == 2
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

@inline function init!(dpf::DepolField, g::AbstractIsingGraph)
    return init!(dpf, boundlayer(dpf, g))
end

@inline function init!(dpf::DepolField, layer::AbstractIsingLayer)
    dpf.dpf[] = get_dpf(dpf, layer)
    return dpf
end

@inline function _calculate(::ΔH, dpf::DepolField, layer::AbstractIsingLayer, proposal)
    j = at_idx(proposal)
    T = eltype(layer)
    ΔM = delta(proposal)
    D = dpf.dpf[]/dpf.surface_NxNy
    M = dpf.M[]
    ΔD = zero(T)
    if j ∈ dpf # In surface
        z = layers_deep(j, dpf)
        ΔD = -T(dpf.zfunc(z)) * ΔM / dpf.surface_NxNy
    end
    c = dpf.c[]
    return -(D * ΔM + M * ΔD + ΔD * ΔM) / c
end

@inline function _update!(::Metropolis, dpf::DepolField, layer::AbstractIsingLayer, proposal)
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
