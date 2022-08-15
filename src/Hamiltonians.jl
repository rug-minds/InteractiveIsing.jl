""" Hamiltonians"""

# No weights
function HFunc(g::AbstractIsingGraph,idx, state = g.state[idx])::Float32
    
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -g.state[connIdx(conn)]
    end

    return efactor
end

# No weights but magfield
function HMagFunc(g::AbstractIsingGraph,idx, state = g.state[idx])::Float32
    
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -g.state[connIdx(conn)]
    end

    return efactor -g.d.mlist[idx]
end

# When there's weights
function HWeightedFunc(g::AbstractIsingGraph,idx, state = g.state[idx])::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor
end

# Weights and magfield
function HWMagFunc(g::AbstractIsingGraph,idx,state = g.state[idx])::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor -g.d.mlist[idx]
end

