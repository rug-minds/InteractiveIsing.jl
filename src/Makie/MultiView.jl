struct MultiLayerLayout
    rowlines::Vector{Int32}
    collines::Vector{Int32}

end

function createMultipleLayerView(ml, g)
    f = fig(ml)
    simulation = sim(g)

    padding = 50

    state_obs = StateObs[]
    mp = midpanel(ml)
    mp[:obs] = state_obs

    ims = Image[]

    ax = Axis(midpanel(ml)[][1,2], xrectzoom = false, yrectzoom = false)

    y_midpoint = floor.( Int, size(g[1]) ./ 2)

    sizehint!(state_obs, nlayers(simulation)[])
    sizehint!(ims, nlayers(simulation)[])

    sizes = [size(g[i]) for i in 1:(nlayers(simulation)[])]

    for (idx, layer) in enumerate(layers(g))
        push!(state_obs, Observable(state(layer)))
        push!(ims, image!(ax, state_obs[end], colormap = :thermal, fxaa = false, interpolate = false))
        y_midpoint_l = floor.( Int, size(g[idx]) ./ 2)
        ytranslate = y_midpoint[1] - y_midpoint_l[1]
        translate!(ims[end], sum(first.(sizes)[1:(idx-1)]) + (idx-1)*padding, ytranslate)
    end

    on(events(ax.scene).mousebutton) do buttons
        MDrawCircle(buttons, simulation)
        return
    end

    if haskey(etc(ml), :timedfunctions_timer)
        close(etc(ml)[:timedfunctions_timer])
    end
    etc(ml)[:timedfunctions_timer] = Timer((timer) -> notify.(state_obs) ,0., interval = 1/60)
    
    return ax
end