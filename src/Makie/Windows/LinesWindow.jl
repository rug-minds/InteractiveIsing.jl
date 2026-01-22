"""
Create a new Makie window with a lines plot for the observables x and y
    Kwarg:
        fps: frames per second
"""
function async_lines_window(func; fps = 30, lifetime, kwargs...)

    process = Process(func; lifetime , kwargs...)
    Processes.preparedata!(process)
    (;x, y) = getcontext(process)
    w = axis_window(window_type = :Lines)
  
    # Storage of data
    w[:xs] = [x]
    w[:ys] = [y]

    # Need to be view, because async update
    # causes x and y to be different lengths sometimes
    xob = Observable((@view x[1:end]))
    yob = Observable((@view y[1:end]))
    newest_obs = [xob,yob]

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
        Processes.syncclose(process)
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
        Processes.syncclose(process)
        close.(w.timers)
        createtask!(process)
        (;x,y) = getcontext(process)
       
        xob = Observable(@view x[1:end])
        yob = Observable(@view y[1:end])
        newest_obs[1] = xob
        newest_obs[2] = yob
         
        Processes.spawntask!(process)
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

    # If a process is given, add the process controls
    if !isnothing(process)
        w[:process] = process
        on(window_open(w)) do x
            if !x
                quit(process)
            end
        end

        on(w[:pausebutton].clicks) do _
            if w[:paused][]
                pause(process)
            else
                start(process)
            end
        end
    end

    Processes.spawntask!(process) 
    return w
end

function get_data(w::MakieWindow{:Lines})
    return (x = w[:xs], y = w[:ys])
end
export get_data

function get_plot(w::MakieWindow{:Lines})
    display(w.f)
end
export get_plot

change_context!(w::MakieWindow{:Lines}; context...) = changecontext!(w[:process]; context...)

function sync_lines_window(proc1, func; fps = 30, lifetime, kwargs...)

    process = Process(func; lifetime, graphproc = proc1, kwargs...)
    Processes.preparedata!(process)
    (;x, y) = getcontext(process)
    w = axis_window(window_type = :Lines)
  
    # Storage of data
    w[:xs] = [x]
    w[:ys] = [y]

    # Need to be view, because async update
    # causes x and y to be different lengths sometimes
    xob = Observable((@view x[1:end]))
    yob = Observable((@view y[1:end]))
    newest_obs = [xob,yob]

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
        Processes.syncclose(process)
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
        Processes.syncclose(process)
        close.(w.timers)
        createtask!(process)
        (;x,y) = getcontext(process)
       
        xob = Observable(@view x[1:end])
        yob = Observable(@view y[1:end])
        newest_obs[1] = xob
        newest_obs[2] = yob
         
        Processes.spawntask!(process)
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

    # If a process is given, add the process controls
    if !isnothing(process)
        w[:process] = process
        on(window_open(w)) do x
            if !x
                quit(process)
            end
        end

        on(w[:pausebutton].clicks) do _
            if w[:paused][]
                pause(process)
            else
                start(process)
            end
        end
    end

    Processes.spawntask!(process) 
    return w
end
