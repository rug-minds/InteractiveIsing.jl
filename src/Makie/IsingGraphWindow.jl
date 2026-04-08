function IsingGraphWindow(g::IsingGraph)
    w = nothing
    if ndims(g) != 3
        w = axis3d_window()
    else
        w = axis_window()
    end
    ax = w[:ax]
end


function singleView(window)
    ml = window[:layout]
    g = window[:graph]
    mp = midpanel(ml)
    f = fig(ml)
    # simulation = sim(g)

    ml[:current_view] = singleView

    # ISING IMAGE
    
    # mp[:sv_img_ob] = img_ob = Observable(getSingleViewImg(window))
    # mp[:showbfield] = Observable(false)
    # if !haskey(mp, :obs)
    #     mp[:img_obs] = img_obs = Observable(getSingleViewImg(window))
    # else
    #     img_obs = mp[:img_obs]
    # end
    mp[:img_obs] = img_obs = Observable(getSingleViewImg(window))
    window[:image] = img_obs
    mp[:axis_size] = size(img_obs[])
    obs_funcs = etc(ml)[:obs_funcs_singleView] = ObserverFunction[]
    

    # LAYER SELECTOR  BUTTONS
    toppanel(ml)[:sb] = selector_buttons = GridLayout(toppanel(ml)[:mid_grid][3,1], tellwidth = false)
    toppanel(ml)[:sll] = selected_layer_label = lift((x,y) -> "$x/$y", window[:layer_idx], nlayers(g))
    push!(obs_funcs, selected_layer_label.inputs...)

    toppanel(ml)[:sb_l] = selector_buttons[1,1] = leftbutton = Button(f, label = "<", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    toppanel(ml)[:sb_label] = selector_buttons[1,2] = Label(f, selected_layer_label, fontsize = 18)
    toppanel(ml)[:sb_r] = selector_buttons[1,3] = rightbutton = Button(f, label = ">", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    # rowsize!(_grid[1,1].layout, 1, 80)

    # BFIELD BUTTON
    push!(obs_funcs, on(midpanel(ml)[:showbfield].active, weak = true) do _
        img_obs[] = getSingleViewImg(window)
    end)

    push!(obs_funcs, on(leftbutton.clicks, weak = true) do _
        # setLayerIdx!(simulation, layerIdx(simulation)[] -1)
        if window[:layer_idx][] > 1
            window[:layer_idx][] -= 1
        end
    end)

    push!(obs_funcs, on(rightbutton.clicks, weak = true) do _
        if window[:layer_idx][] < nlayers(g)
            window[:layer_idx][] += 1
        end
    end)

    push!(obs_funcs, on(window[:layer_idx], weak = true) do _
        setLayerSV(window)
    end)


    cur_layer = current_layer(window)

    # Create the axis for the layer type
    create_layer_axis!(window, mp, pos = (1,2))

    ax = mp[:axis]
    
    # mp[:axis].yreversed[] = @load_preference("makie_y_flip", default = false)
    # mp[:image].colorrange[] = (-1,1)

    #TODO:Restore this
    # push!(obs_funcs, on(events(ax.scene).mousebutton, weak = true) do buttons
    #     # MDrawCircle(ax, buttons, simulation)
    #     drawCircle(state(current_layer(window)), tuple(round.(Int, mouseposition(ax.scene))...), 20)
    #     return
    # end)


    # me_axis = addmouseevents!(ax.scene)

    # onmouseleftdown(me_axis) do ev
    #     println("Dragging stopped")
    #     mp[:me_axis] = me_axis
    #     return
    # end
    # onmouseleftdragstop(me_axis) do _
    #     println("Dragging stopped")
    #     # close(mp["mousedrag_timer"])
    #     return
    # end

    # Weightgenerator display
    wg_label_obs = lift(x-> "$(wg(g[x]))", window[:layer_idx])
    push!(obs_funcs, wg_label_obs.inputs...)
    bp = bottompanel(ml)
    bp_midgrid_toprow = 1+bp[:mid_grid].offsets[1]

    bottompanel(ml)[:wf_label] = Label(bp[:mid_grid][bp_midgrid_toprow - 1,1], wg_label_obs, fontsize = 12)

    # TIMER FOR THE SCREEN
    if haskey(ml, :timedfunctions_timer)
        close(ml[:timedfunctions_timer])
    end

    # iob = mp[:img_obs]
    # pushmainfunc!(window, (window) -> notify(iob))
    pushmainfunc!(window, (window) -> notify(mp[:img_obs]))

    return
end