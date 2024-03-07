using GLMakie, InteractiveIsing
set_window_config!(;
        vsync = false,
        framerate = 60.0,
        pause_renderloop = false,
        focus_on_show = true,
        decorated = true,
        title = "Monte Carlo"
    )
    
function estimatePi()
    set_window_config!(;
        vsync = false,
        framerate = 60.0,
        pause_renderloop = false,
        focus_on_show = true,
        decorated = true,
        title = "Monte Carlo"
    )

    f, screen, isopen = empty_window(resolution = (2000, 1000))
    grid = GridLayout(f[1,1], halign = :left)
    points_in = Observable(Point2f[])
    points_out = Observable(Point2f[])
    num_in = Observable(0)
    num_out = Observable(0)
    pi_est = Observable(0.)
    
    labelgrid = GridLayout(grid[1,2])

    pilabel = Label(labelgrid[1,1], "π = 4*n_in/n", fontsize = 36, halign = :left)

    n_in_text = lift((n_in) -> "n_in = $n_in", num_in)
    n_out_text = lift((n_out) -> "n_out = $n_out", num_out)
    n_inlabel = Label(labelgrid[2,1], n_in_text, tellheight = false, fontsize = 36, halign = :left)
    n_outlabel = Label(labelgrid[2,2], n_out_text, tellheight = false, fontsize = 36, halign = :left)
    # colsize!(grid,2, 100)

    label_text = lift((est) -> "π ≈ $(round(est, digits = 7))", pi_est)
    lab = Label(labelgrid[3,1], label_text, tellheight = false, fontsize = 36, halign = :left)
    run = Ref(true)

    ax = Axis(f[1,1], aspect = 1, halign = :left)
    arc!(ax, Point2f(0,0), 1, 0, pi/2)
    scatter!(ax, points_in, color = :blue)
    scatter!(ax, points_out, color = :red)
    resize_to_layout!(f)

    function loop()
        @async while run[]
            p = rand(Point2f)
            is_in = p[1]^2 + p[2]^2 < 1
            if is_in
                num_in.val += 1
                push!(points_in[], p)
            else
                num_out.val += 1
                push!(points_out[], p)
            end
            yield()
            GC.safepoint()
        end
    end

    function update()
        pi_est[] = 4*num_in[]/(num_in[]+num_out[])
        notify(points_in)
        notify(points_out)
        notify(num_in)
        notify(num_out)
    end

    on(isopen) do x
        if x
            run[] = true
            loop()
        else
            run[] = false
        end
    end
    loop()
    timer = create_window_timer(update, isopen)
    return pi_est
end