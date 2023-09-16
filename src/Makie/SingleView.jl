export create_singleview
max_r_slider(simulation, slider) = slider.range[] = 1:(floor(Int64,min(size(currentLayer(simulation))...)/2))
sim_max_r(simulation) = (floor(Int64,min(size(currentLayer(simulation))...)/2))
"""
"""
function create_singleview(ml, g)
    mp = midpanel(ml)
    f = fig(ml)
    simulation = sim(g)
    poplist = etc(ml)["poplist"]

    # LAYER SELECTOR  BUTTONS
    selector_buttons = GridLayout(toppanel(ml)[][3,2], tellwidth = false)
    selected_layer_label = lift((x,y) -> "$x/$y", layerIdx(simulation), nlayers(simulation))
    # Pop this listener when the label is removed
    push!(poplist, layerIdx(simulation))
    push!(poplist, nlayers(simulation))

    selector_buttons[1,2] = Label(f, selected_layer_label, fontsize = 18)
    selector_buttons[1,1] = leftbutton = Button(f, label = "<", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    selector_buttons[1,3] = rightbutton = Button(f, label = ">", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    # rowsize!(_grid[1,1].layout, 1, 80)

    # ISING IMAGE
    img_ob = Observable{Base.ReshapedArray}(state(currentLayer(simulation)))
    # img_ob = Observable{Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int32}}, true}, Tuple{}}}(state(currentLayer(simulation)))

    # max_r_slider(simulation, midpanel(ml)["rslider"]) 

    on(leftbutton.clicks) do _
        changeLayer(-1, simulation)
        img_ob[] = state(currentLayer(simulation))
        # max_r_slider(simulation, midpanel(ml)["rslider"]) 
        reset_limits!(ax)
    end
    on(rightbutton.clicks) do _
        changeLayer(1, simulation)
        img_ob[] = state(currentLayer(simulation))
        # max_r_slider(simulation, midpanel(ml)["rslider"]) 
        reset_limits!(ax)
    end
    # ax = Axis(mp[][1,2], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), ypanlock = true, xpanlock = true, yzoomlock = true, xzoomlock = true)
    ax = Axis(mp[][1,2], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    im = image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)

    # rowsize!(_grid, 1, Relative(1/20))

    mp["axis"] = ax
    mp["image"] = im
    mp["obs"] = img_ob

    on(events(ax.scene).mousebutton) do buttons
        MDrawCircle(ax, buttons, simulation)
        return
    end
    # me_axis = addmouseevents!(ax.scene)

    # onmouseleftdown(me_axis) do ev
    #     println("Dragging stopped")
    #     mp["me_axis"] = me_axis
    #     return
    # end
    # onmouseleftdragstop(me_axis) do _
    #     println("Dragging stopped")
    #     # close(mp["mousedrag_timer"])
    #     return
    # end

    # Weightgenerator display
    wf_label_obs = lift(x-> "$(wg(g[x]))", layerIdx(simulation))
    push!(poplist, layerIdx(simulation))
    bg = bottompanel(ml)["bottomgrid"]
    bottompanel(ml)["wf_label"] = Label(bg[0,1], wf_label_obs, fontsize = 18)

    if haskey(etc(ml), "timedfunctions_timer")
        close(etc(ml)["timedfunctions_timer"])
    end

    etc(ml)["timedfunctions_timer"] = Timer((timer) -> (notify(mp["obs"]); timedFunctions(simulation)) ,0., interval = 1/60)

    return
end
