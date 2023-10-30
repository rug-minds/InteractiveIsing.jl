function totalEnergy(graph)
    total_E = 0f0
    for layer in layers(graph)
        total_E += totalEnergy(layer)
    end
    return total_E
end

function totalEnergy(layer::IsingLayer)
    g = graph(layer)
    _state = copy(state(g))
    adj = sp_adj(g)
    _stype = stype(g)
    total_E = 0f0
    for state_idx in graphidxs(layer)
        total_E += _state[state_idx] * getdE(g, _state, adj, state_idx, _stype)
    end
    return total_E
end
export totalEnergy