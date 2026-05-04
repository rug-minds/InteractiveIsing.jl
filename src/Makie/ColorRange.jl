const DEFAULT_UNBOUNDED_COLORRANGE = (-1.0f0, 1.0f0)

function layer_colorrange(layer)
    range = stateset(layer)
    if isfinite(range[1]) && isfinite(range[end]) && range[1] < range[end]
        return (range[1], range[end])
    end

    T = eltype(state(layer))
    return (T(DEFAULT_UNBOUNDED_COLORRANGE[1]), T(DEFAULT_UNBOUNDED_COLORRANGE[2]))
end

function bind_layer_colorrange!(plot, state_obs, layer)
    plot.colorrange[] = layer_colorrange(layer)
    return plot
end
