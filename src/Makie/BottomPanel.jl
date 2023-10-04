function bottomPanel(ml, g)
    f = fig(ml)
    simulation = sim(g)

    # BOTTOM PANEL
    bg = GridLayout(f[3,1], tellheight = false, tellwidth = false)
    bp = bottompanel(ml, LayoutPanel(bg))
    bp["bottomgrid"] = bg
    
    bp["mid_grid"] = mid_grid = GridLayout(bg[1,1], tellwidth = false)
    # Magnetization label for layer
    bp["m_text"] = m_text = lift(x -> "Magnetization: $x", M(simulation))
    # Pop this listener when the label is removed
    bp["m_label"] = mid_grid[1,1] = Label(f, m_text, fontsize = 18)
    bp["p_textbox"] = p_textbox = UIntTextbox(mid_grid[2,1], 
        onfunc = (num) -> addRandomDefects!(currentLayer(simulation), num), 
        placeholder = "% Defect", 
        width = 100
    )

    rowsize!(f.layout, 3, 140)    
end

function cleanup(ml, ::typeof(bottomPanel))
    bottompanel(ml, LayoutPanel())
end

