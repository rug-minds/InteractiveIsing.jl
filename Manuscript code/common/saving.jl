function result_basename(p::ManuscriptParams)
    d = derived_params(p)
    date_str = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    return string(
        "Scale=", round(p.Scale, digits = 4),
        "_Screening=", round(p.Screening, digits = 4),
        "_timefctr=", round(p.time_fctr, digits = 4),
        "_Steps_1=", round(p.Steps_1, digits = 4),
        "_Eb=", round(d.E_barrier, digits = 4),
        "_Epp=", round(d.Epp_1, digits = 4),
        "_Temp_aneal=", round(p.Temp_aneal, digits = 4),
        "_", date_str,
    )
end

function state_distribution(g; bins = -1.5:0.05:1.5)
    P = graph_array(g)
    h = fit(Histogram, vec(P), bins)
    density = h.weights ./ sum(h.weights)
    return (; histogram = h, density, bins)
end

function save_distribution_figure(path, dist)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "P", ylabel = "Probability")
    barplot!(ax, dist.histogram.edges[1][1:end-1], dist.density; width = step(dist.bins))
    save(path, fig)
    return fig
end

function save_series_figure(path; x, y, xlabel, ylabel = "Pr", title = "")
    n = min(length(x), length(y))
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel, ylabel, title)
    lines!(ax, Float64.(x[1:n]), Float64.(y[1:n]))
    save(path, fig)
    return fig
end

function save_landau_figure(path, p::ManuscriptParams; xrange = range(-1.5, 1.5, length = 1000))
    coeffs = landau_coefficients(p)
    energy = [landau_energy(coeffs, x) for x in xrange]

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Pr", ylabel = "Landau energy", title = "Landau energy")
    lines!(ax, Float64.(collect(xrange)), Float64.(energy))
    save(path, fig)
    return fig
end

function excel_decimal(x)
    return replace(string(Float64(x)), "." => ",")
end

function save_final_state_csv(path, g)
    A = graph_array(g)
    open(path, "w") do io
        println(io, "x,y,z,value")
        @inbounds for z in axes(A, 3), y in axes(A, 2), x in axes(A, 1)
            println(io, x, ",", y, ",", z, ",", Float64(A[x, y, z]))
        end
    end
    return path
end

function save_final_state_csv_eu(path, g)
    A = graph_array(g)
    open(path, "w") do io
        println(io, "sep=;")
        println(io, "x;y;z;value")
        @inbounds for z in axes(A, 3), y in axes(A, 2), x in axes(A, 1)
            println(io, x, ";", y, ";", z, ";", excel_decimal(A[x, y, z]))
        end
    end
    return path
end

function save_final_state_figure(path, g; vmin = -1.5, vmax = 1.5)
    A = graph_array(g)
    nx, ny, nz = size(A)
    n = nx * ny * nz
    xs = Vector{Float32}(undef, n)
    ys = Vector{Float32}(undef, n)
    zs = Vector{Float32}(undef, n)
    cs = Vector{Float32}(undef, n)

    k = 1
    @inbounds for z in 1:nz, y in 1:ny, x in 1:nx
        xs[k] = x
        ys[k] = y
        zs[k] = z
        cs[k] = A[x, y, z]
        k += 1
    end

    fig = Figure(size = (1000, 800))
    ax = Axis3(fig[1, 1]; xlabel = "x", ylabel = "y", zlabel = "z",
        aspect = (1, 1, 1), azimuth = 1.15, elevation = 0.35, title = "Final state")
    scatter!(ax, xs, ys, zs; color = cs, colormap = [:red, :black],
        colorrange = (vmin, vmax), markersize = 10)
    Colorbar(fig[1, 2]; colormap = [:red, :black], colorrange = (vmin, vmax), label = "P")
    save(path, fig)
    return fig
end

excel_value(x::Union{Missing, Bool, Float64, Int64, Dates.Date, Dates.DateTime, Dates.Time, String}) = x
excel_value(x::Integer) = Int64(x)
excel_value(x::AbstractFloat) = Float64(x)
excel_value(x::Symbol) = String(x)
excel_value(x) = repr(x)

function save_run_outputs(g, p::ManuscriptParams; anneal = nothing, pulse = nothing)
    mkpath(p.outdir)
    base_name = result_basename(p)
    xlsx_path = joinpath(p.outdir, base_name * ".xlsx")
    png_path_dist = joinpath(p.outdir, base_name * "_Pr_distribution.png")
    png_path_pv = joinpath(p.outdir, base_name * "_PV.png")
    png_path_pt = joinpath(p.outdir, base_name * "_PT.png")
    png_path_landau = joinpath(p.outdir, base_name * "_Landau.png")
    png_path_landau_zoom = joinpath(p.outdir, base_name * "_Landau_zoom_m1_1.png")
    csv_path_state = joinpath(p.outdir, base_name * "_FinalState_standard.csv")
    csv_path_state_eu = joinpath(p.outdir, base_name * "_FinalState_excel_eu.csv")
    png_path_state = joinpath(p.outdir, base_name * "_FinalState.png")

    dist = state_distribution(g)
    save_distribution_figure(png_path_dist, dist)
    save_landau_figure(png_path_landau, p)
    save_landau_figure(png_path_landau_zoom, p; xrange = range(-1.0, 1.0, length = 1000))
    save_final_state_csv(csv_path_state, g)
    save_final_state_csv_eu(csv_path_state_eu, g)
    save_final_state_figure(png_path_state, g)

    if !isnothing(pulse)
        save_series_figure(
            png_path_pv;
            x = pulse.voltage,
            y = pulse.Pr,
            xlabel = "Voltage",
            ylabel = "Pr",
            title = "P-V",
        )
    end

    if !isnothing(anneal)
        save_series_figure(
            png_path_pt;
            x = anneal.Temp,
            y = anneal.Pr,
            xlabel = "Temperature",
            ylabel = "Pr",
            title = "P-T",
        )
    end

    d = derived_params(p)
    bin_left = Float64.(dist.histogram.edges[1][1:end-1])
    bin_center = bin_left .+ step(dist.bins) / 2
    df_dist = DataFrame(
        bin_left = bin_left,
        bin_center = bin_center,
        prob = Float64.(dist.density),
        counts = Float64.(dist.histogram.weights),
    )

    coeffs = d.landau_coeffs
    param_keys = String[
        "JIsing", "a1", "b1", "c1", "E_barrier", "Eypp_1", "xL", "yL", "zL",
        "Scale", "Screening", "Steps_1", "time_fctr", "anneal_time",
        "point_repeat", "Temp_aneal", "landau_mode", "landau_coeffs",
        "landau_2", "landau_4", "landau_6", "landau_8", "landau_10",
    ]
    param_values = Any[
        p.JIsing, p.a1, d.b1, p.c1, d.E_barrier, d.Epp_1, p.xL, p.yL, p.zL,
        p.Scale, p.Screening, p.Steps_1, p.time_fctr, d.anneal_time,
        d.point_repeat, p.Temp_aneal, p.landau_mode, d.landau_coeffs,
        get(coeffs, 2, missing), get(coeffs, 4, missing), get(coeffs, 6, missing),
        get(coeffs, 8, missing), get(coeffs, 10, missing),
    ]

    params = DataFrame(
        key = param_keys,
        value = excel_value.(param_values),
    )

    XLSX.openxlsx(xlsx_path, mode = "w") do xf
        xf[1].name = "params"
        XLSX.writetable!(xf["params"], collect(eachcol(params)), names(params))

        XLSX.addsheet!(xf, "Pr_distribution")
        XLSX.writetable!(xf["Pr_distribution"], collect(eachcol(df_dist)), names(df_dist))

        if !isnothing(anneal)
            n = min(length(anneal.Temp), length(anneal.Pr))
            df_anneal = DataFrame(Temp = Float64.(anneal.Temp[1:n]), Pr = Float64.(anneal.Pr[1:n]))
            XLSX.addsheet!(xf, "anneal_series")
            XLSX.writetable!(xf["anneal_series"], collect(eachcol(df_anneal)), names(df_anneal))
        end

        if !isnothing(pulse)
            n = min(length(pulse.voltage), length(pulse.Pr))
            df_pulse = DataFrame(voltage = Float64.(pulse.voltage[1:n]), Pr = Float64.(pulse.Pr[1:n]))
            XLSX.addsheet!(xf, "pulse_series")
            XLSX.writetable!(xf["pulse_series"], collect(eachcol(df_pulse)), names(df_pulse))
        end
    end

    return (; xlsx_path, png_path_dist, png_path_landau, png_path_landau_zoom,
        csv_path_state, csv_path_state_eu, png_path_state,
        png_path_pv = isnothing(pulse) ? nothing : png_path_pv,
        png_path_pt = isnothing(anneal) ? nothing : png_path_pt)
end
