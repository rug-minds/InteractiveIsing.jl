"""
Get a list of the coordinates of the outgoing connections of a node in a graph.
"""
function conn_coords(g,i)
    idxToCoord.(graph(g).adj.rowval[nzrange(graph(g).adj,i)],Ref(size(g)))
end

function conn_idxs(g,i)
    graph(g).adj.rowval[nzrange(graph(g).adj,i)]
end

function conn_weights(g,i)
    graph(g).adj.nzval[nzrange(graph(g).adj,i)]
end