
# DEV:
export ml, midpanel, toppanel, bottompanel

function baseFig(window; framerate = 30.0)
    ml = window[:layout]
    g = window[:graph]
    if haskey(ml, :basefig_active) && ml[:basefig_active]
        cleanup(ml, baseFig)
    end

    timedFunctions[:upf] = updatesPerFrame
    timedFunctions[:magnetization] = magnetization
    # sim.timers[:makie] = PTimer((timer) -> timerFuncs(sim) ,0., interval = interval)


    ml[:basefig_active] = true

    # f = fig(ml, Figure(size = (1500, 1500)))
    f = window.f

    GLMakie.activate!(;
        vsync = false,
        framerate = framerate,
        pause_renderloop = false,
        focus_on_show = true,
        decorated = true,
        title = "Interactive Ising Simulation"
    )

    topPanel(window)

    midPanel(window)

    bottomPanel(window)

    # if disp
    #     println("Displaying")
    #     screen = display(f)
    #     ml[:screen] = screen
    # end

    return f
end

function cleanup(ml, ::typeof(baseFig))
    currentview = ml[:current_view]
    if !isnothing(currentview)
        cleanup(ml, currentview)
    end

    try 
        # close(simulation[].timers[:makie])
        # delete!(simulation[].timers, :makie)
    catch
    end
    
    delete!(timedFunctions, :upf)
    delete!(timedFunctions, :magnetization)

    cleanup(ml, topPanel)
    cleanup(ml, midPanel)
    cleanup(ml, bottomPanel)
    return nothing
end

