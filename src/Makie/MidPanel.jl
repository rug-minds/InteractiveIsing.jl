function create_midpanel(ml,g)
    f = fig(ml)
    simulation = sim(g)

    poplist = etc(ml)["poplist"]

    # Mid Panel
    midgrid = GridLayout(f[2,1])
    midpanel(ml, LayoutPanel(midgrid))
    mp = midpanel(ml)

    # LEFT Panel
    mp["leftpanel"] = leftpanel = GridLayout(midgrid[1,1], tellheight = false)

            # LEFT PANEL 1,1
            mp["buttons"] = buttons = GridLayout(leftpanel[1,1])

            # Brush buttons
                mp["bs"] = bs = [Button(buttons[-i+2,1], padding = (0,0,0,0), fontsize = 24, width = 40, height = 40, label = "$i") for i in 1:-1:-1]

                for (idx,val) in enumerate(1:-1:-1)
                    on(bs[idx].clicks) do _
                        # println("clicked $val")
                        brush(simulation)[] = Float32(val)
                    end
                end

            # Clamp toggle
                mp["clamplabel"] = Label(buttons[4,1], "Clamping", fontsize = 18)

                mp["clamptoggle"] = clamptoggle = Toggle(buttons[5,1], active = false)

            # LEFT PANEL 2,2
            # SIZE TEXTBOX 
                mp["sizetextbox"] = size_grid = GridLayout(leftpanel[2,1])

                size_validator(r_string) = try 0 < parse(UInt, r_string) < sim_max_r(simulation); catch; false; end
                mp["size_label_text"] = sl_text = lift(x -> "Radius < $(sim_max_r(simulation))", layerIdx(simulation)) 
                mp["sizelabel"] = Label(size_grid[1,1], sl_text)
                mp["sizefield"] = sizefield = Textbox(size_grid[2,1], placeholder = " ", validator = size_validator, width = 40)
                sizefield.stored_string[] = sizefield.displayed_string[] = string(sim_max_r(simulation))


                on(brushR(simulation)) do x
                    if mp["sizefield"].stored_string[] != string(x)
                        mp["sizefield"].stored_string[] = string(x)
                    end
                end

                on(sizefield.stored_string) do s
                    if s != string(brushR(simulation)[])
                        brushR(simulation)[] = parse(UInt, s)
                    end 
                end

            # SHOW BFIELD
            mp["showbfield"] = showbfield = Toggle(leftpanel[4,1], active = false)
            mp["showbfieldlabel"] = Label(leftpanel[3,1], "Show BField", fontsize = 18)

            on(showbfield.active) do x
                if x
                    midpanel(ml)["sv_img_ob"][] = bfield(currentLayer(simulation))
                else
                    midpanel(ml)["sv_img_ob"][] = state(currentLayer(simulation))
                end
            end
        

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
    mp["rightpanel"] = rightpanel = GridLayout(midgrid[1,3])

        # TEMP SLIDER
            mp["temptext"] = lift(x -> "T: $x", Temp(simulation))
            # Pop this listener when the label is removed
            push!(poplist, Temp(simulation))


            Box(rightpanel[1,1], width = 100, height = 50, visible = false)
            mp["templabel"] = Label(rightpanel[1,1], mp["temptext"], fontsize = 18)
            tempslider = mp["tempslider"] = tempslider = Slider(rightpanel[2,1], range = 0.0:0.1:20, value = 1.0, horizontal = false)
            
            on(tempslider.value) do value
                Temp(simulation)[] = value
            end
        
    # rowsize!(f.layout, 1, Auto())
    colsize!(mp[], 1, 200)
    # colsize!(mp[], 3, 100)


    
end