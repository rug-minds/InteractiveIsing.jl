function midPanel(window)
    ml = window[:layout]
    g = window[:graph]
    f = fig(ml)
    # simulation = sim(g)
    window[:brush_r] = Observable(sim_max_r(window)/2)

    obs_funcs = ml[:obs_funcs_midPanel] = ObserverFunction[]
    coupled_obs = ml[:coupled_obs_midPanel] = Observable[]

    # Mid Panel
    midgrid = GridLayout(f[2,1])
    midpanel(ml, LayoutPanel(midgrid))
    mp = midpanel(ml)

    # LEFT Panel
    mp[:leftpanel] = leftpanel = GridLayout(midgrid[1,1], tellheight = false)

            # LEFT PANEL 1,1
            mp[:buttons] = buttons = GridLayout(leftpanel[1,1])

            # Brush buttons
                mp[:bs] = bs = [Button(buttons[-i+2,1], padding = (0,0,0,0), fontsize = 24, width = 40, height = 40, label = "$i") for i in 1:-1:-1]

                for (idx,val) in enumerate(1:-1:-1)
                    push!(obs_funcs, on(bs[idx].clicks) do _
                        # println("clicked $val")
                        brush(simulation)[] = Float32(val)
                    end)
                end

            # Clamp toggle
                mp[:clamplabel] = Label(buttons[4,1], "Clamping", fontsize = 18)

                mp[:clamptoggle] = clamptoggle = Toggle(buttons[5,1], active = false)

            # LEFT PANEL 2,2
            # SIZE TEXTBOX 
                mp[:sizetextbox] = size_grid = GridLayout(leftpanel[2,1])

                size_validator(r_string) = try 0 < parse(UInt, r_string) < sim_max_r(window); catch; false; end
                mp[:size_label_text] = sl_text = lift(x -> "Radius < $(sim_max_r(window))", window[:layer_idx])
                mp[:sizelabel] = Label(size_grid[1,1], sl_text)
                mp[:sizefield] = sizefield = UIntTextbox(size_grid[2,1], 
                    onfunc = (num) -> window[:brush_r][] = num,
                    placeholder = "$(window[:brush_r][])",
                    upper = () -> sim_max_r(window), 
                    width = 40, 
                    defocus_on_submit = true, 
                    reset_on_defocus = true)

                # TODO: Restore
                push!(obs_funcs, on(window[:brush_r]) do x
                    sizefield.placeholder[] = string(x)
                    #Set circle
                    circ(simulation, getOrdCirc(window[:brush_r][]))
                end)

            # SHOW BFIELD
            mp[:showbfield] = showbfield = Toggle(leftpanel[4,1], active = false)
            mp[:showbfieldlabel] = Label(leftpanel[3,1], "Show BField", fontsize = 18)

            push!(obs_funcs, on(showbfield.active) do x
                mp[:img_obs][] = getSingleViewImg(window)
            end)
        

            # SIZE SLIDER
                # leftpanel[1,2] = slidergrid = GridLayout(tellheight = false)

                # mp["rslider_text"] = lift(x -> "r: $x", brushR(simulation))
                # # Pop this listener when the label is removed
                # push!(poplist, brushR(simulation))

                # Box(slidergrid[1,1], width = 100, height = 50, visible = false)

                # mp["rsliderlabel"] = slidergrid[1,1] = Label(f, mp["rslider_text"] , fontsize = 18, width = 32, tellwidth = false)

                # mp["rslider"] = rslider = Slider(slidergrid[2,1], range = 1:100, value = 20, horizontal = false)


                # on(rslider.value) do x
                #     brushR(simulation)[] = Int32(rslider.value[])
                # end

                # # on(value_dragstop(rslider)) do _
                # #     circ(simulation, getOrdCirc(brushR(simulation)[]))
                # # end

                # # Initialize r and circle
                # value_dragstop(rslider)[] = rslider.value[] = floor(Int64, min(size(currentLayer(simulation))...)/8)


    # RIGHT PANEL
    mp[:rightpanel] = rightpanel = GridLayout(midgrid[1,3])

        # TEMPERATURE SLIDER
            # push!(coupled_obs, mp["temptext"])
            # Pop this listener when the label is removed
            


            Box(rightpanel[1,1], width = 100, height = 50, visible = false)
            tempslider = mp[:tempslider] = tempslider = Slider(rightpanel[2,1], range = 0.0:0.02:20, value = 1.0, horizontal = false)
            
            tempslider.value.ignore_equal_values = true

            set_close_to!(tempslider, temp(g)[])
            window[:gtemp] = PolledObservable(temp(g), (o) -> temp(g))
            on(window[:gtemp]) do x
                temp(g, x)
                set_close_to!(tempslider, x)
            end
            pushpolled!(window, window[:gtemp])
            ob_pair = Observables.ObservablePair(tempslider.value, window[:gtemp])
            mp[:ob_pair] = ob_pair

            #Isn't this redundant?
            on(tempslider.value) do x
                set_close_to!(tempslider, x)
            end

            push!(obs_funcs, ob_pair.links...)

            mp[:temptext] = lift(x -> "T: $x", tempslider.value)

            mp[:templabel] = Label(rightpanel[1,1], mp[:temptext], fontsize = 18)
        
    # rowsize!(f.layout, 1, Auto())
    colsize!(mp[], 1, 200)
    # colsize!(mp[], 3, 100)
end

function cleanup(ml, ::typeof(midPanel))
        @justtry if !isempty(ml[:obs_funcs_midPanel])
            off.(ml[:obs_funcs_midPanel])
        end
        @justtry delete!(ml, :obs_funcs_midPanel)
        # decouple!.(ml["coupled_obs_midPanel"])
        @justtry delete!(ml, :coupled_obs_midPanel)

        midpanel(ml, LayoutPanel())
end