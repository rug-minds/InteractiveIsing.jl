createAvgWindow(layer; buffersize = 512, window_time = 1) = createAvgWindow(layer; buffersize, framerate = buffersize*window_time)
function createAvgWindow(layer; buffersize = 512, framerate = 256)
    ml = mlref[]
    if haskey(ml, "avgwindow_active") && ml["avgwindow_active"]
        cleanup(ml, avgWindow)
    end
    avgWindow(ml, layer, buffersize, framerate)
end

closeAvgWindow() = cleanup(mlref[], avgWindow)

export createAvgWindow, closeAvgWindow

struct avgWindow{T} <: Function
    l::IsingLayer
    buffers::Matrix{AverageCircular{T}}
    img_ob::Observable{Matrix{T}}
    screen::GLMakie.Screen
    timer::Timer
    other::Dict{String, Any}
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
    ax = Axis(f[1, 1], aspect = 1)
    
    newscreen = GLMakie.Screen()
    timer = Timer(t -> update(buffers, img_ob, layer), 0, interval = 1/framerate)


    avgwindow = avgWindow(layer, buffers, img_ob, newscreen, timer, Dict{String, Any}())
    
    ml["avgWindow"] = avgwindow

    # DISPLAY
    d = display(newscreen, f)

    # Make image
    image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    
    push!(ml.cleanuplist, avgWindow)

end

function update(buffers, img_ob, l)
    t = Threads.@spawn begin
        update_buffers(buffers, state(l))
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
    avgWindow = ml["avgWindow"]
    close(avgWindow.timer)
    GLFW.SetWindowShouldClose(to_native(avgWindow.screen), true)
    return nothing
end