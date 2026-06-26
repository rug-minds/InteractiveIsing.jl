function result_basename(p::ManuscriptParams)
    date_str = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    short_id = randstring('A':'Z', 4) * randstring('0':'9', 2)
    return "Basefile_$(date_str)_$(short_id)"
end

function state_distribution(g; bins = -1.5:0.05:1.5)
    P = graph_array(g)
    h = fit(Histogram, vec(P), bins)
    total = sum(h.weights)
    density = iszero(total) ? zeros(Float64, length(h.weights)) : h.weights ./ total
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

_landau_is_scalar_coeff(coeff) = coeff isa Number
_landau_has_spatial_coeffs(coeffs) = any(!_landau_is_scalar_coeff(coeff) for coeff in values(coeffs))

function _mean_landau_coefficients(coeffs)
    return Dict(
        Int(order) => (
            coeff isa Number ? Float64(coeff) : Float64(sum(coeff) / length(coeff))
        )
        for (order, coeff) in pairs(coeffs)
    )
end

function _spatial_landau_energy_band(coeffs, xrange)
    flattened = Dict{Int,Vector{Float64}}()
    nsites = nothing

    for (order, coeff) in pairs(coeffs)
        coeff isa Number && continue
        values_flat = Float64.(vec(coeff))
        if isnothing(nsites)
            nsites = length(values_flat)
        else
            length(values_flat) == nsites || error("All array-valued Landau coefficient fields must have the same size.")
        end
        flattened[Int(order)] = values_flat
    end

    isnothing(nsites) && error("Spatial Landau energy band requires at least one array-valued coefficient.")

    xs = Float64.(collect(xrange))
    mins = Vector{Float64}(undef, length(xs))
    maxs = Vector{Float64}(undef, length(xs))
    means = Vector{Float64}(undef, length(xs))
    temp = Vector{Float64}(undef, nsites)

    for (i, x) in pairs(xs)
        fill!(temp, 0.0)
        for (order, coeff) in pairs(coeffs)
            power = x^order
            if coeff isa Number
                temp .+= Float64(coeff) * power
            else
                temp .+= flattened[Int(order)] .* power
            end
        end
        mins[i] = minimum(temp)
        maxs[i] = maximum(temp)
        means[i] = sum(temp) / nsites
    end

    return (; xs, mins, maxs, means)
end

function save_landau_figure(path, p::ManuscriptParams; xrange = range(-1.5, 1.5, length = 1000))
    coeffs = landau_coefficients(p)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Pr", ylabel = "Landau energy", title = "Landau energy")

    if _landau_has_spatial_coeffs(coeffs)
        band = _spatial_landau_energy_band(coeffs, xrange)
        lines!(ax, band.xs, band.means, color = :blue)
        band!(ax, band.xs, band.mins, band.maxs, color = (:blue, 0.18))
    else
        energy = [landau_energy(coeffs, x) for x in xrange]
        lines!(ax, Float64.(collect(xrange)), Float64.(energy))
    end

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

function numeric_or_missing_column(values, nrows)
    column = Vector{Union{Missing,Float64}}(missing, nrows)
    isnothing(values) && return column
    n = min(length(values), nrows)
    n > 0 && (column[1:n] .= Float64.(values[1:n]))
    return column
end

function run_series_dataframe(; anneal = nothing, pulse = nothing)
    columns = Pair{Symbol,Any}[]
    function add_result!(prefix, result)
        isnothing(result) && return
        for name in propertynames(result)
            name == :context && continue
            values = getproperty(result, name)
            isnothing(values) && continue
            values isa AbstractVector || continue
            push!(columns, Symbol(prefix, "_", name) => values)
        end
    end
    add_result!("anneal", anneal)
    add_result!("pulse", pulse)
    isempty(columns) && return DataFrame()
    nrows = maximum(length(last(pair)) for pair in columns)
    return DataFrame((first(pair) => numeric_or_missing_column(last(pair), nrows) for pair in columns))
end

function params_dataframe(p::ManuscriptParams)
    return DataFrame(
        key = String.(propertynames(p)),
        value = [excel_value(getproperty(p, name)) for name in propertynames(p)],
    )
end

function derived_dataframe(p::ManuscriptParams)
    d = derived_params(p)
    return DataFrame(
        key = String.(propertynames(d)),
        value = [excel_value(getproperty(d, name)) for name in propertynames(d)],
    )
end

function coefficient_summary_dataframe(p::ManuscriptParams)
    rows = NamedTuple[]
    for (order, coeff) in sort(collect(landau_coefficients(p)); by = first)
        values = coeff isa Number ? [Float64(coeff)] : Float64.(vec(coeff))
        push!(rows, (;
            order,
            coefficient = "P$(order)",
            mean = mean(values),
            std = length(values) > 1 ? std(values) : 0.0,
            minimum = minimum(values),
            maximum = maximum(values),
            first_value = first(values),
            n = length(values),
        ))
    end
    return DataFrame(rows)
end

function excel_cell_value(x)
    ismissing(x) && return missing
    x isa Bool && return x
    x isa Integer && return Int64(x)
    x isa Real && return Float64(x)
    x isa Dates.Date && return x
    x isa Dates.DateTime && return x
    x isa Dates.Time && return x
    x isa AbstractString && return String(x)
    return string(x)
end

function write_dataframe_sheet!(ws, df::DataFrame)
    if ncol(df) == 0
        ws["A1"] = "empty"
        return nothing
    end
    for (j, name) in enumerate(names(df))
        ws[XLSX.CellRef(1, j)] = string(name)
    end
    for i in 1:nrow(df), j in 1:ncol(df)
        ws[XLSX.CellRef(i + 1, j)] = excel_cell_value(df[i, j])
    end
    return nothing
end

function save_diagnostic_figures(base_name, outdir; anneal = nothing, pulse = nothing)
    paths = String[]
    specs = Pair{String,NamedTuple}[]
    if !isnothing(anneal)
        push!(specs, "anneal_T_Pr" => (; x = anneal.Temp, y = anneal.Pr,
            xlabel = "Temperature", ylabel = "Pr", title = "P-T"))
    end
    if !isnothing(pulse)
        push!(specs, "pulse_V_Pr" => (; x = pulse.voltage, y = pulse.Pr,
            xlabel = "Voltage", ylabel = "Pr", title = "P-V"))
        for (suffix, field, ylabel) in (
            ("pulse_Htotal_V", :H_total, "Total H"),
            ("pulse_Hrest_V", :H_rest, "H_dep + H_J + H_poly"),
            ("pulse_PAFEz_V", :P_AFE_z, "P_AFE_z"),
        )
            values = getproperty(pulse, field)
            isnothing(values) && continue
            push!(specs, suffix => (; x = pulse.voltage, y = values,
                xlabel = "Voltage", ylabel, title = suffix))
        end
    end
    for (suffix, spec) in specs
        path = joinpath(outdir, base_name * "_" * suffix * ".png")
        save_series_figure(path; spec...)
        push!(paths, path)
    end
    return paths
end

function save_run_outputs(
    g,
    p::ManuscriptParams;
    anneal = nothing,
    pulse = nothing,
    base_name = result_basename(p),
    save_figures = p.save_figures,
    save_xlsx = p.save_xlsx,
)
    mkpath(p.outdir)
    xlsx_path = joinpath(p.outdir, base_name * ".xlsx")
    png_path_dist = joinpath(p.outdir, base_name * "_Pr_distribution.png")
    png_path_pv = joinpath(p.outdir, base_name * "_pulse_V_Pr.png")
    png_path_pt = joinpath(p.outdir, base_name * "_anneal_T_Pr.png")
    png_path_landau = joinpath(p.outdir, base_name * "_Landau.png")
    png_path_landau_zoom = joinpath(p.outdir, base_name * "_Landau_zoom_m1_1.png")
    csv_path_state = joinpath(p.outdir, base_name * "_FinalState_standard.csv")
    csv_path_state_eu = joinpath(p.outdir, base_name * "_FinalState_excel_eu.csv")
    png_path_state = joinpath(p.outdir, base_name * "_FinalState.png")

    dist = state_distribution(g)
    save_final_state_csv(csv_path_state, g)
    save_final_state_csv_eu(csv_path_state_eu, g)
    figure_paths = String[]
    if save_figures
        save_distribution_figure(png_path_dist, dist)
        save_landau_figure(png_path_landau, p)
        save_landau_figure(png_path_landau_zoom, p; xrange = range(-1.0, 1.0, length = 1000))
        save_final_state_figure(png_path_state, g; vmin = p.state_min, vmax = p.state_max)
        append!(figure_paths, (png_path_dist, png_path_landau, png_path_landau_zoom, png_path_state))
        append!(figure_paths, save_diagnostic_figures(base_name, p.outdir; anneal, pulse))
    end

    bin_left = Float64.(dist.histogram.edges[1][1:end-1])
    bin_center = bin_left .+ step(dist.bins) / 2
    df_dist = DataFrame(
        bin_left = bin_left,
        bin_center = bin_center,
        prob = Float64.(dist.density),
        counts = Float64.(dist.histogram.weights),
    )

    if save_xlsx
        sheets = (
            "series" => run_series_dataframe(; anneal, pulse),
            "params" => params_dataframe(p),
            "derived" => derived_dataframe(p),
            "reduced_energy" => reduced_parameter_summary(g, p),
            "landau_coefficients" => coefficient_summary_dataframe(p),
            "Pr_distribution" => df_dist,
        )
        XLSX.openxlsx(xlsx_path, mode = "w") do xf
            for (i, (name, df)) in enumerate(sheets)
                if i == 1
                    xf[1].name = name
                else
                    XLSX.addsheet!(xf, name)
                end
                write_dataframe_sheet!(xf[name], df)
            end
        end
    end

    return (; xlsx_path, png_path_dist, png_path_landau, png_path_landau_zoom,
        csv_path_state, csv_path_state_eu, png_path_state,
        figure_paths,
        png_path_pv = isnothing(pulse) ? nothing : png_path_pv,
        png_path_pt = isnothing(anneal) ? nothing : png_path_pt)
end
