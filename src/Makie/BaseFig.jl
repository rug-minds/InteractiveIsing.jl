
# DEV:
export ml, midpanel, toppanel, bottompanel

function baseFig(g; disp = true)
    ml = mlref[]
    if haskey(ml, "basefig_active") && ml["basefig_active"]
        cleanup(ml, baseFig)
    end


    ml["basefig_active"] = true

    f = fig(ml, Figure(resolution = (1500, 1500)))

    set_window_config!(;
        vsync = false,
        framerate = 60.0,
        pause_renderloop = false,
        focus_on_show = true,
        decorated = true,
        title = "Interactive Ising Simulation"
    )

    topPanel(ml, g)

    midPanel(ml,g)

    bottomPanel(ml, g)

    if disp
        println("Displaying")
        screen = display(f)
        ml["screen"] = screen
    end

    return f
end

function cleanup(ml, ::typeof(baseFig))
    currentview = ml["current_view"]
    if !isnothing(currentview)
        cleanup(ml, currentview)
    end

    cleanup(ml, topPanel)
    cleanup(ml, midPanel)
    cleanup(ml, bottomPanel)
    return nothing
end

