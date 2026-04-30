function _run_capture_dir(p::ManuscriptParams)
    name = "Scale=$(round(p.Scale, digits = 4))_Screening=$(round(p.Screening, digits = 4))"
    return joinpath(p.outdir, name, "capture")
end

"""
    start_pulse_sweep(paramsets; repeats = 1)

Build all graphs and start all pulse processes first. Call `fetch_pulse_sweep`
afterwards to wait for results. This avoids serializing the sweep by fetching
inside the launch loop.
"""
function start_pulse_sweep(paramsets; repeats = 1, run_builder = build_pulse_process)
    runs = Any[]
    for p in paramsets
        g = build_graph(p)
        run = run_builder(g, p; capture_dir = _run_capture_dir(p))
        process = start_pulse!(g, run; repeats)
        push!(runs, (; params = p, graph = g, run, process))
    end
    return runs
end

function fetch_pulse_sweep(runs)
    return map(runs) do item
        result = fetch_run(item.process, item.run)
        (; item.params, item.graph, item.run, item.process, result)
    end
end

export start_pulse_sweep, fetch_pulse_sweep
