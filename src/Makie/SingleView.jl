export singleView
max_r_slider(simulation, slider) = slider.range[] = 1:(floor(Int64,min(size(currentLayer(simulation))...)/2))
function sim_max_r(simulation)
    maxsize = (floor(Int64,min(size(currentLayer(simulation))...)/2))
    return maxsize == 0 ? 1 : maxsize
end

export flip_y_axis
"""
Create a view of the graph that shows a single layer at the time
cycling through layers is possible with the < and > buttons
"""
function singleView(ml, g)
    mp = midpanel(ml)
    f = fig(ml)
    simulation = sim(g)

    ml["current_view"] = singleView

    # ISING IMAGE
    
    mp["sv_img_ob"] = img_ob = Observable{Base.ReshapedArray}(state(currentLayer(simulation)))
    # img_ob[] = getSingleViewImg(g, ml) #Bfield or state
    mp["axis_size"] = size(img_ob[])
    obs_funcs = etc(ml)["obs_funcs_singleView"] = ObserverFunction[]
    # coupled_obs = etc(ml)["coupled_obs_singleView"] = Observable[]
    

    # LAYER SELECTOR  BUTTONS
    toppanel(ml)["sb"] = selector_buttons = GridLayout(toppanel(ml)["mid_grid"][3,1], tellwidth = false)
    toppanel(ml)["sll"] = selected_layer_label = lift((x,y) -> "$x/$y", layerIdx(simulation), nlayers(simulation))
    push!(obs_funcs, selected_layer_label.inputs...)

    toppanel(ml)["sb_<"] = selector_buttons[1,1] = leftbutton = Button(f, label = "<", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    toppanel(ml)["sb_label"] = selector_buttons[1,2] = Label(f, selected_layer_label, fontsize = 18)
    toppanel(ml)["sb_>"] = selector_buttons[1,3] = rightbutton = Button(f, label = ">", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    # rowsize!(_grid[1,1].layout, 1, 80)

    # BFIELD BUTTON
    push!(obs_funcs, on(midpanel(ml)["showbfield"].active, weak = true) do _
        img_ob[] = getSingleViewImg(g, ml)
    end)
    # img_ob = Observable{Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int32}}, true}, Tuple{}}}(state(currentLayer(simulation)))

    

    push!(obs_funcs, on(leftbutton.clicks, weak = true) do _
        setLayerIdx!(simulation, layerIdx(simulation)[] -1)
    end)

    push!(obs_funcs, on(rightbutton.clicks, weak = true) do _
        setLayerIdx!(simulation, layerIdx(simulation)[] + 1)
    end)

    push!(obs_funcs, on(layerIdx(simulation), weak = true) do val
        setLayerSV(val)
    end)


    cur_layer = currentLayer(simulation)

    # Create the axis for the layer type
    create_layer_axis!(cur_layer, mp, pos = (1,2))

    ax = mp["axis"]
    
    # mp["axis"].yreversed[] = @load_preference("makie_y_flip", default = false)
    # mp["image"].colorrange[] = (-1,1)

    push!(obs_funcs, on(events(ax.scene).mousebutton, weak = true) do buttons
        MDrawCircle(ax, buttons, simulation)
        return
    end)
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
    wg_label_obs = lift(x-> "$(wg(g[x]))", layerIdx(simulation))
    push!(obs_funcs, wg_label_obs.inputs...)
    bp = bottompanel(ml)
    bp_midgrid_toprow = 1+bp["mid_grid"].offsets[1]
    bottompanel(ml)["wf_label"] = Label(bp["mid_grid"][bp_midgrid_toprow - 1,1], wg_label_obs, fontsize = 12)

    # TIMER FOR THE SCREEN
    if haskey(ml, "timedfunctions_timer")
        close(ml["timedfunctions_timer"])
    end

    timedFunctions["screen"] = (sim) -> notify(mp["obs"])

    return
end
create_singleview = singleView

using GLMakie.Makie.GridLayoutBase: deleterow!
function cleanup(ml, ::typeof(singleView))
    # Observables
    off.(etc(ml)["obs_funcs_singleView"])
    delete!(ml, "obs_funcs_singleView")
    # decouple!.(ml["coupled_obs_singleView"])
    delete!(ml, "coupled_obs_singleView")


    # Selector buttons
    sb = toppanel(ml)["sb"]
    for idx in 1:length(sb.content)
        c = first(sb.content).content
        delete!(c)
    end

    tp = toppanel(ml)
    deleterow!(tp["sb"].parent, 3)
    delete!(tp, "sb", "sb_<", "sb_label", "sb_>")

    # Axis
    mp = midpanel(ml)
    delete!(mp["axis"])
    delete!(mp, "axis", "image")

    # Weightgenerator display
    bp = bottompanel(ml)
    delete!(bp["wf_label"])
    delete!(bp, "wf_label")
    bp_midgrid_toprow = 1+bp["mid_grid"].offsets[1] 
    deleterow!(bp["mid_grid"], bp_midgrid_toprow)

    #Timer
    # close(ml["timedfunctions_timer"])
    delete!(timedFunctions, "screen")

    delete!(etc(ml), "timedfunctions_timer")
    if !isnothing(simulation)
        close(timers(simulation[])["makie"])
        delete!(timers(simulation[]), "makie")
    end
    

    ml["current_view"] = nothing
    

end
export cleanup