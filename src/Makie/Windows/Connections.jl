export visualize_connections
function visualize_connections(g, i)
    window = new_window()
    # Make GridLayout
    window["gridlayout"] = window.f[1, 1] = GridLayout()

    colorvec = zeros(RGBAf, nstates(g))
    weights = conn_weights(g,i)
    colorvec[conn_idxs(g,i)] .= weight_colors(weights)
    # return (colorvec)
    window["ax"] = create_layer_axis!(g, window, color = colorvec)
end

visualize_connections(g, i, j) = visualize_connections(g, coordToIdx(Int32.((i, j)), size(g)))
visualize_connections(g, i, j, k) = visualize_connections(g, coordToIdx(Int32.((i, j, k)), size(g)) )

function weight_colors(weights)
    max_weight = maximum(weights)
    min_weight = minimum(weights)
    colorvec = zeros(RGBAf, length(weights))
    for w_idx in eachindex(weights)
        #linear interpolation from purple to yellow
        colorvec[w_idx] = RGBAf(0.5, 0.0, 0.5 + 0.5*(weights[w_idx] - min_weight)/(max_weight - min_weight), 0.1 + 0.9*(weights[w_idx] - min_weight)/(max_weight - min_weight))
    end
    return colorvec
end