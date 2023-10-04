function topPanel(ml, g)
    f = fig(ml)
    simulation = sim(g)

    obs_funcs = ml["obs_funcs_topPanel"] = ObserverFunction[]
    coupled_obs = ml["coupled_obs_topPanel"] = Observable[]

    topgrid = GridLayout(f[1,1])
    tp = toppanel(ml, LayoutPanel(topgrid))

    # TOP BUTTONS

        # Create an invisible box
        toppanel(ml)["topbox"] = topbox = Box(topgrid[1,1:3], visible = false)
        # leftbox = Box(topgrid[1:2,1], visible = true)
        colsize!(topgrid, 1, Relative(1/3))
        colsize!(topgrid, 2, Relative(1/3))
        colsize!(topgrid, 3, Relative(1/3))



        tp["mid_grid"] = mid_grid = GridLayout(topgrid[1,2], tellwidth = false)
        tp["resetbutton"] = resetbutton = Button(mid_grid[1,1], label = "Reset Graph", fontsize = 18, height = 30, halign = :center, tellwidth = false)
        
        tp["buttontext"] = buttontext = liftcouple(x -> x ? "Paused" : "Running", isPaused(simulation))
        push!(coupled_obs, tp["buttontext"])

        tp["pausebutton"] = pausebutton = Button(mid_grid[2,1], padding = (0,0,0,0), fontsize = 18, width = 100, height = 30, label = buttontext, halign = :center, tellwidth = false)
        
        push!(obs_funcs, on(resetbutton.clicks) do _
            reset!(simulation)
        end)

        push!(obs_funcs, on(pausebutton.clicks) do _
            togglePause(g)
        end)

    # UPDATE COUNTERS

        tp["upsgrid"] = upsgrid = GridLayout(topgrid[1,1])

        update_label_upf = liftcouple(x -> "Updates per frame: $x", upf(simulation))
        push!(coupled_obs, update_label_upf)


        update_label_upfps = liftcouple(x -> "Updates per frame per spins: $x", upfps(simulation))
        push!(coupled_obs, update_label_upfps)
        
        tp["updates_label"] = upsgrid[1,1] = updates_label = Label(f, update_label_upf, padding = (0,0,0,0), fontsize = 18, halign = :left, valign = :top, tellheight = false, tellwidth = false)
        tp["updates_label_per_spin"] = upsgrid[2,1] = updates_label_per_spin = Label(f, update_label_upfps, fontsize = 18,  halign = :left, valign = :top, tellheight = false, tellwidth = false)
    # TOP RIGHT PANEL

    # Etcetera buttons
    etc_bs = GridLayout(topgrid[1,3], halign = :right)

    # Saving Image
    img = rotr90(load(modulefolder*"/Makie/Icons/cam.png"))
    cam_a = ImageAxis(etc_bs[1,1], width = 30, height = 25, halign = :right, valign = :top, tellwidth = false)
    tp["cam_a"] = cam_a
    tp["cam_b"] = image!(cam_a, img)

    me = addmouseevents!(f.scene, cam_a.layoutobservables.computedbbox)
    push!(obs_funcs, onmouseleftclick(me) do _
        saveGImg(currentLayer(simulation))
    end)

    # Plot correlation function
    corr_b = Button(etc_bs[2,1], label = "Correlation", padding = (0,0,0,0), fontsize = 14, width = 80, height = 30, halign = :right, valign = :top, tellwidth = false)
    push!(obs_funcs, on(corr_b.clicks) do _
        plotCorr(correlationLength(currentLayer(simulation)), save = false)
    end)
end

function cleanup(ml, ::typeof(topPanel))
    off.(ml["obs_funcs_topPanel"])
    delete!(ml, "obs_funcs_topPanel")
    decouple!.(ml["coupled_obs_topPanel"])
    delete!(ml, "coupled_obs_topPanel")
    
    toppanel(ml, LayoutPanel())
end