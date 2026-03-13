function createAvgWindow(layer; buffersize = 512, window_time = 1, framerate = buffersize*window_time)
    ml = simulation[].ml[]
    if haskey(ml, :avgwindow_active) && ml[:avgwindow_active]
        cleanup(ml, avgWindow)
    end
    avgWindow(ml, layer, buffersize, framerate)
end

closeAvgWindow() = cleanup(simulation[].ml[], avgWindow)

export createAvgWindow, closeAvgWindow

mutable struct AvgWindow{T} <: Function
    l::IsingLayer
    buffers::Matrix{AverageCircular{T}}
    img_ob::Observable{Matrix{T}}
    screen::GLMakie.Screen
    f::Figure
    other::Dict{Symbol, Any}
    timer::Timer
    AvgWindow(a,b,c,d,e,f) = new{eltype(c[])}(a,b,c,d,e,f)
end

function avgWindow(ml, layer, buffersize, framerate)
    g = graph(layer)
    buffertype = eltype(g)
    _size = size(layer)
    buffers = Matrix{AverageCircular{buffertype}}(undef, _size...)
    img_ob = Observable(zeros(buffertype, _size...))

    for idx in eachindex(buffers)
        buffers[idx] = AverageCircular(buffertype, buffersize)
    end

    f = Figure();
    newscreen = GLMakie.Screen()
    display(newscreen, f)


    ax = Axis(f[1, 1], aspect = 1)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    

    avgwindow = AvgWindow(layer, buffers, img_ob, newscreen, f, Dict{Symbol, Any}())
    ml[:avgWindow] = avgwindow
    
    # DISPLAY

    # Make image
    image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    
    push!(ml.cleanuplist, avgWindow)

    # timedFunctions["avgWindow"] = update_avgWindow

    on(events(ml[:avgWindow].f).window_open) do _
        cleanup(ml, avgWindow)
    end
    
    avgwindow.timer = Timer((timer) -> update_avgWindow(simulation[]) ,0., interval = 1/framerate)
end

function update_avgWindow(sim)
    ml = sim.ml[]::SimLayout
    avgWindow = ml[:avgWindow]
    buffers = avgWindow.buffers
    img_ob = avgWindow.img_ob

    t = Threads.@spawn begin
        update_buffers(buffers, state(avgWindow.l))
        update_img(img_ob, buffers)
    end
    wait(t)
    notify(img_ob)
end

function update_buffers(buffers, state)
    for idx in eachindex(buffers)
        push!(buffers[idx], state[idx])
    end
end

function update_img(img_ob, buffers)
    avgs = img_ob.val
    for idx in eachindex(avgs)
        avgs[idx] = avg(buffers[idx])
    end
end

function cleanup(ml, ::typeof(avgWindow))
    println("Cleaning up avgWindow")
    avgwindow = ml[:avgWindow]
    close(avgwindow.timer)
    # delete!(timedFunctions, "avgWindow")
    try 
        GLFW.SetWindowShouldClose(to_native(avgwindow.screen), true)
    catch
    end
    delete!(ml, :avgWindow)
    return nothing
end