function create_bottompanel(ml, g)
    f = fig(ml)
    simulation = sim(g)

    poplist = etc(ml)["poplist"]

    # BOTTOM PANEL
    bg = GridLayout(f[3,1], tellheight = false, tellwidth = false)
    bp = bottompanel(ml, LayoutPanel(bg))
    bp["bottomgrid"] = bg
    
    # Magnetization label for layer
    bp["m_text"] = m_text = lift(x -> "Magnetization: $x", M(simulation))
    # Pop this listener when the label is removed
    push!(poplist, M(simulation))
    val(x) = try 0 < parse(UInt64,x) < 100; catch; false; end
    bp["m_label"] = bg[1,1] = Label(f, m_text, fontsize = 18)
    bp["p_textbox"] = bg[2,1] = p_textbox = Textbox(f, placeholder = "% Defect", validator = val, defocus_on_submit = true, reset_on_defocus = true, width = 100)
    
    on(p_textbox.stored_string) do s
        if s != nothing
            p = parse(UInt64, s)
            addRandomDefects!(currentLayer(simulation), p)
            p_textbox.stored_string[] = nothing
        end
    end

    rowsize!(f.layout, 3, Relative(1/8))    
end