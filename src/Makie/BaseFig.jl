const ml = MakieLayout(Figure())

# DEV:
export ml, midpanel, toppanel, bottompanel

function createBaseFig(g, create_view  = (a,b) -> nothing; disp = true)
    f = fig(ml, Figure(resolution = (1500, 1500)))

    set_window_config!(;
        vsync = false,
        framerate = 60.0,
        pause_renderloop = false,
        focus_on_show = true,
        decorated = true,
        title = "Interactive Ising Simulation"
    )


    # Delete all observables from last fig
    if haskey(etc(ml), "poplist")
        for idx in eachindex(etc(ml)["poplist"])
            pop!(etc(ml)["poplist"][idx].listeners)
        end
    end

    etc(ml)["poplist"] = Observable[]

    create_toppanel(ml, g)

    create_midpanel(ml,g)

    create_bottompanel(ml, g)

    create_view(ml, g)

    if disp
        println("Displaying")
        display(f)
    end

    return f
end
export createBaseFig