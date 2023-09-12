const ml = MakieLayout(Figure())

# DEV:
export ml, midpanel, toppanel, bottompanel

function createBaseFig(g, create_view  = (a,b) -> nothing)
    f = fig(ml, Figure(resolution = (1200, 1500)))

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
    
    display(f)

    return f
end
export createBaseFig