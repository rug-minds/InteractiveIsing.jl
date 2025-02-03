function BarWindow(func; fps = 30, lifetime, kwargs...)

    proc = Process(func; lifetime , kwargs...)
    preparedata!(proc)
    (;x, y) = getargs(proc)
    # xref = Ref(x)
    # yref = Ref(y)
    w = axis_window(window_type = :Lines)
    # x = xref[]
    # y = yref[]
    
    # first = true
    # Storage of data
    w[:xs] = [x]
    w[:ys] = [y]

    # Need to be view, because async update
    # causes x and y to be different lengths sometimes
    xob = Observable((@view x[1:end]))
    yob = Observable((@view y[1:end]))

    newest_obs = [xob,yob]
    # xob = Observable(@view x[1:end])
    # yob = Observable(@view y[1:end])

    lines = lines!(w[:ax], xob, yob)
    w[:lines] = [lines]

    function timerfunc(timer)
        # The timer updates async, so somtimes it might update the
        # Plot just in between an x and y update. So, it needs
        # To use views that are set to the same size at the moment
        # Of updating the screen
        minlength = min(length(newest_obs[1][].parent), length(newest_obs[2][].parent))
        # Silent update both
        newest_obs[1].val = @view newest_obs[1][].parent[1:minlength]
        newest_obs[2].val = @view newest_obs[2][].parent[1:minlength]

        notify(newest_obs[1])
        autolimits!(w[:ax])
    end
    
    # pushtimer!(w,PTimer((timer) -> begin notify(x); autolimits!(w[:ax]); end, 0., interval = 1/fps))
    pushtimer!(w, PTimer(timerfunc, 0., interval = 1/fps))
    
    reset() = begin
        Processes.syncclose(proc)
        for line in w[:lines]
            delete!(w[:ax], line)
        end
        deleteat!(w[:lines], 1:length(w[:lines]))
        deleteat!(w[:xs], 1:length(w[:xs]))
        deleteat!(w[:ys], 1:length(w[:ys]))
        first = true
    end
    # Reset Button
    resetbutton = Button(w.f[0,1][1,2], label = "Reset", tellwidth = false, height = 28)
    on(resetbutton.clicks) do _
        reset()
    end

    function newlines!()
        Processes.syncclose(proc)
        close.(w.timers)
        createtask!(proc)
        (;x,y) = getargs(proc)
       
        xob = Observable(@view x[1:end])
        yob = Observable(@view y[1:end])
        newest_obs[1] = xob
        newest_obs[2] = yob
         
        spawntask!(proc)
        start.(w.timers)
        push!(w[:lines], lines!(w[:ax], xob, yob))
    end

    #Rerun Button
    rerunbutton = Button(w.f[0,1][1,3], label = "Rerun", tellwidth = false, height = 28)
    on(rerunbutton.clicks) do _
            newlines!()
            # Start process
            w[:paused][] = false
    end

    # Space to reset
    on(events(w.f.scene).keyboardbutton) do event
        if event.action == Keyboard.press
            if event.key == Keyboard.space
                reset()
            end
        end
    end

    # If a proc is given, add the proc controls
    if !isnothing(proc)
        w[:proc] = proc
        on(window_open(w)) do x
            if !x
                quit(proc)
            end
        end

        on(w[:pausebutton].clicks) do _
            if w[:paused][]
                pause(proc)
            else
                start(proc)
            end
        end
    end

    spawntask!(proc) 
    return w
end