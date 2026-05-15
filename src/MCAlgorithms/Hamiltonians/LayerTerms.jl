export LayerTerm, layeridx, boundlayer, _calculate, _update!

"""
    LayerTerm

Hamiltonian term whose parameters and internal state are defined on one graph
layer. Concrete layer terms store an integer `layer` selector at construction
time. Graph and layer references are resolved only during instantiation and
calculation.
"""
abstract type LayerTerm <: HamiltonianTerm end

"""
    _calculate(functional, term, model_or_layer, args...)

Internal implementation hook for Hamiltonian terms.

Ordinary terms may implement either `calculate` directly or `_calculate` through
the generic fallback. `LayerTerm`s should usually implement `_calculate` on the
bound layer; the public `calculate` wrapper handles scope checks and index
translation.
"""
function _calculate end

"""
    _update!(algo, term, layer, proposal)

Internal update hook for scoped layer terms. The default is a no-op.
"""
_update!(algo, term, model, proposal) = nothing

@inline function calculate(hF::AbstractLinearFunctional, hterm::HamiltonianTerm, model, args...)
    return @inline _calculate(hF, hterm, model, args...)
end

@inline function _calculate(hF::AbstractLinearFunctional, hterm::HamiltonianTerm, model, args...)
    throw(MethodError(_calculate, (hF, hterm, model, args...)))
end

@inline layeridx(term::LayerTerm) = getfield(term, :layer)

@inline function boundlayer(term::LayerTerm, model::AbstractIsingGraph)
    idx = layeridx(term)
    1 <= idx <= length(model) ||
        throw(BoundsError(model, idx))
    return model[idx]
end

@inline boundlayer(::LayerTerm, layer::AbstractIsingLayer) = layer

@inline _layerterm_zero(model) = zero(eltype(model))

@inline in_bound_layer(term::LayerTerm, proposal::FlipProposal) =
    getfield(proposal, :layer_idx) == layeridx(term)

@inline function in_bound_layer(term::LayerTerm, model::AbstractIsingGraph, spin_idx::Integer)
    layer = @inline boundlayer(term, model)
    return spin_idx in graphidxs(layer)
end

@inline function local_spin_idx(term::LayerTerm, model::AbstractIsingGraph, spin_idx::Integer)
    layer = @inline boundlayer(term, model)
    return @inline idxGToL(spin_idx, layer)
end

@inline local_spin_idx(::LayerTerm, ::AbstractIsingLayer, spin_idx::Integer) = spin_idx

@inline function localproposal(term::LayerTerm, model::AbstractIsingGraph, proposal::FlipProposal)
    local_idx = @inline local_spin_idx(term, model, at_idx(proposal))
    return FlipProposal(
        proposal,
        local_idx,
        from_val(proposal),
        to_val(proposal),
        1,
        isaccepted(proposal),
    )
end

@inline localproposal(::LayerTerm, ::AbstractIsingLayer, proposal::FlipProposal) = proposal

@inline function calculate(hF::H, term::LayerTerm, model::AbstractIsingGraph)
    layer = @inline boundlayer(term, model)
    return @inline _calculate(hF, term, layer)
end

@inline function calculate(hF::ΔH, term::LayerTerm, model::AbstractIsingGraph, proposal::FlipProposal)
    (@inline in_bound_layer(term, proposal)) || return @inline _layerterm_zero(model)
    layer = @inline boundlayer(term, model)
    local_fp = @inline localproposal(term, model, proposal)
    return @inline _calculate(hF, term, layer, local_fp)
end

@inline function calculate(hF::AbstractLinearFunctional, term::LayerTerm, model::AbstractIsingGraph, spin_idx::Integer)
    (@inline in_bound_layer(term, model, spin_idx)) || return @inline _layerterm_zero(model)
    layer = @inline boundlayer(term, model)
    local_idx = @inline local_spin_idx(term, model, spin_idx)
    return @inline _calculate(hF, term, layer, local_idx)
end

@inline function update!(algo, term::LayerTerm, model::AbstractIsingGraph, proposal::FlipProposal)
    isaccepted(proposal) || return nothing
    (@inline in_bound_layer(term, proposal)) || return nothing
    layer = @inline boundlayer(term, model)
    local_fp = @inline localproposal(term, model, proposal)
    return @inline _update!(algo, term, layer, local_fp)
end

function _layerterm_constructor(::Type{H}) where {H<:LayerTerm}
    return Base.typename(H).wrapper
end

function LayerTerm(::Type{H}, layer::Integer, params, internals) where {H<:LayerTerm}
    hasparams = _hasfield(H, :parameters)
    hasinternal = _hasfield(H, :internal)
    constructor = _layerterm_constructor(H)

    if hasparams && hasinternal
        return constructor(Int(layer), params, internals)
    elseif hasparams
        return constructor(Int(layer), params)
    elseif hasinternal
        return constructor(Int(layer), internals)
    else
        throw(ArgumentError("LayerTerm type $(H) must define `parameters` and/or `internal` fields."))
    end
end

function instantiate(term::H, model::AbstractIsingGraph) where {H<:LayerTerm}
    layer = @inline boundlayer(term, model)
    return LayerTerm(
        H,
        layeridx(term),
        instantiate(parameters(term), layer),
        instantiate(internal(term), layer),
    )
end

# Minimal graph-like interface for using an `AbstractIsingLayer` as the
# template-instantiation model for a `LayerTerm`.
@inline statelen(layer::AbstractIsingLayer) = length(state(layer))
@inline graphstate(layer::AbstractIsingLayer) = state(layer)
@inline function adjGToL(A::UndirectedAdjacency, layer::AbstractIsingLayer)
    idxs = graphidxs(layer)
    local_sp = getfield(A, :sp)[idxs, idxs]
    local_diag = separate_diagonal(A) ? getfield(A, :diag)[idxs] : nothing
    return UndirectedAdjacency(local_sp, local_diag; fastwrite = fastwrite(A))
end

@inline adjGToL(A, layer::AbstractIsingLayer) = A[graphidxs(layer), graphidxs(layer)]
@inline adj(layer::AbstractIsingLayer) = @inline adjGToL(adj(graph(layer)), layer)
