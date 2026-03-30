function init_connection_triplets_from_layers(precision, num_nodes, layers::AbstractLayerData...)
    rows = Int32[]
    cols = Int32[]
    vals = precision[]
    for layer in layers
        layerrows, layercols, layervals = gen_connections(layer, precision, get_weightgenerator(layer), num_nodes)
        append!(rows, layerrows)
        append!(cols, layercols)
        append!(vals, layervals)
    end
    return rows, cols, vals
end

function init_connections_from_layers(precision, num_nodes, layers::AbstractLayerData...)
    rows, cols, vals = init_connection_triplets_from_layers(precision, num_nodes, layers...)
    return sparse(rows, cols, vals, num_nodes, num_nodes)
end
