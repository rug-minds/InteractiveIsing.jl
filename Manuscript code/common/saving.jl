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

    dist = state_distribution(g)
    save_distribution_figure(png_path_dist, dist)
    save_landau_figure(png_path_landau, p)

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

    param_keys = String[
        "JIsing", "a1", "b1", "c1", "E_barrier", "Eypp_1", "xL", "yL", "zL",
        "Scale", "Screening", "Steps_1", "time_fctr", "anneal_time",
        "point_repeat", "Temp_aneal", "landau_mode", "landau_coeffs",
    ]
    param_values = Any[
        p.JIsing, p.a1, d.b1, p.c1, d.E_barrier, d.Epp_1, p.xL, p.yL, p.zL,
        p.Scale, p.Screening, p.Steps_1, p.time_fctr, d.anneal_time,
        d.point_repeat, p.Temp_aneal, p.landau_mode, d.landau_coeffs,
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

    return (; xlsx_path, png_path_dist, png_path_landau,
        png_path_pv = isnothing(pulse) ? nothing : png_path_pv,
        png_path_pt = isnothing(anneal) ? nothing : png_path_pt)
end
