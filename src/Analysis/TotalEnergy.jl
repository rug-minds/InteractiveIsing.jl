function totalEnergy(graph::IsingGraph{T}) where T
    total_E = zero(T)
    for layer in layers(graph)
        total_E += totalEnergy(layer)
    end
    return total_E
end

# Slow because of runtime dispatch
function totalEnergy(layer::IsingLayer)
    g = graph(layer)
    Base.eltype = precision(g)
    _state = copy(state(g))
    adj = adj(g)
    _stype = stype(g)
    total_E = zero(Base.eltype)
    for state_idx in graphidxs(layer)
        total_E += _state[state_idx] * getdE(g, _state, adj, state_idx, _stype)
    end
    return total_E
end
export totalEnergy