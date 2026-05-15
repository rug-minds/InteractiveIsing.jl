export ToLayer, inner

"""
    ToLayer(layer, hamiltonian)

Wrap an ordinary Hamiltonian so it is instantiated and evaluated on one graph
layer as a mini graph. The wrapper uses the generic `LayerTerm` scope checks:
out-of-layer proposals contribute zero and do not update internal caches.
"""
struct ToLayer{H<:Hamiltonian} <: LayerTerm
    layer::Int
    hamiltonian::H
end

ToLayer(layer::Integer, hamiltonian::H) where {H<:Hamiltonian} =
    ToLayer{H}(Int(layer), hamiltonian)

ToLayer(layer::Integer, ::Type{H}) where {H<:Hamiltonian} =
    ToLayer(layer, H())

@inline inner(t::ToLayer) = getfield(t, :hamiltonian)

function instantiate(t::ToLayer, model::AbstractIsingGraph)
    layer = @inline boundlayer(t, model)
    return ToLayer(layeridx(t), instantiate(inner(t), layer))
end

@inline function _calculate(hF::AbstractLinearFunctional, t::ToLayer, layer::AbstractIsingLayer, args...)
    return @inline calculate(hF, inner(t), layer, args...)
end

@inline function _update!(algo, t::ToLayer, layer::AbstractIsingLayer, proposal)
    return @inline update!(algo, inner(t), layer, proposal)
end

function gethamiltonian(t::ToLayer, ::Type{H}) where {H}
    h = inner(t)
    h isa H && return h
    h isa HamiltonianTerms && return gethamiltonian(h, H)
    error("Type $H not found in ToLayer wrapper")
end

function gethamiltonian(t::ToLayer, ::Type{H}, layer::Integer) where {H}
    layeridx(t) == Int(layer) ||
        error("ToLayer is bound to layer $(layeridx(t)), not layer $(Int(layer))")
    return gethamiltonian(t, H)
end

function _try_get_layer_hamiltonian(hterm, ::Type{H}, layer::Integer) where {H}
    if hterm isa ToLayer && layeridx(hterm) == layer
        try
            return gethamiltonian(hterm, H)
        catch
            return nothing
        end
    elseif hterm isa LayerTerm && hterm isa H && layeridx(hterm) == layer
        return hterm
    end
    return nothing
end

function gethamiltonian(hts::HamiltonianTerms, ::Type{H}, layer::Integer) where {H}
    for hterm in hamiltonians(hts)
        found = _try_get_layer_hamiltonian(hterm, H, Int(layer))
        isnothing(found) || return found
    end
    error("Hamiltonian type $H not found on layer $(Int(layer))")
end
