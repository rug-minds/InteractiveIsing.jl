using Dates

"""Return the latency experiment output directory, creating it if needed."""
function latency_outdir(prefix::AbstractString)
    root = get(ENV, "ISING_LATENCY_DIR", joinpath(@__DIR__, "runs"))
    path = joinpath(root, prefix * "_" * Dates.format(now(), "yyyymmdd_HHMMSS"))
    mkpath(path)
    return path
end

"""Measure `f()` and append wall/compile/memory timing fields to `rows`."""
function measure!(f, rows::Vector{Dict{String,Any}}, phase::AbstractString)
    timing = @timed f()
    push!(rows, Dict{String,Any}(
        "phase" => String(phase),
        "seconds" => timing.time,
        "compile_time" => timing.compile_time,
        "recompile_time" => timing.recompile_time,
        "gctime" => timing.gctime,
        "bytes" => timing.bytes,
    ))
    println(
        rpad(phase, 36),
        " seconds=", round(timing.time, digits = 4),
        " compile=", round(timing.compile_time, digits = 4),
        " recompile=", round(timing.recompile_time, digits = 4),
        " gc=", round(timing.gctime, digits = 4),
    )
    return timing.value
end

"""Write rows produced by `measure!` to a compact CSV file."""
function write_latency_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "phase,seconds,compile_time,recompile_time,gctime,bytes")
        for row in rows
            println(io, join((
                row["phase"],
                row["seconds"],
                row["compile_time"],
                row["recompile_time"],
                row["gctime"],
                row["bytes"],
            ), ","))
        end
    end
    return path
end

"""Write one Markdown latency report with settings and measured phase timings."""
function write_latency_md(path::AbstractString, title::AbstractString, settings, rows, notes)
    open(path, "w") do io
        println(io, "# ", title)
        println(io)
        println(io, "Generated: `", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "`")
        println(io)
        println(io, "## Settings")
        println(io)
        for (key, value) in pairs(settings)
            println(io, "- `", key, "`: `", value, "`")
        end
        println(io)
        println(io, "## Phase Timings")
        println(io)
        println(io, "| phase | seconds | compile time | recompile time | gc time | bytes |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for row in rows
            println(
                io,
                "| `", row["phase"], "` | ",
                round(row["seconds"], digits = 4), " | ",
                round(row["compile_time"], digits = 4), " | ",
                round(row["recompile_time"], digits = 4), " | ",
                round(row["gctime"], digits = 4), " | ",
                row["bytes"], " |",
            )
        end
        println(io)
        println(io, "## Notes")
        println(io)
        for note in notes
            println(io, "- ", note)
        end
    end
    return path
end

"""Return a short named tuple containing common profile settings."""
function latency_settings(; kwargs...)
    return (; kwargs...)
end
