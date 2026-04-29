struct TrianglePulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end

function Processes.init(tp::TrianglePulseA, args)
    steps = num_calls(args)
    num_samples = steps / (4 * tp.numpulses)
    first = LinRange(0, tp.amp, round(Int, num_samples))
    second = LinRange(tp.amp, 0, round(Int, num_samples))
    third = LinRange(0, -tp.amp, round(Int, num_samples))
    fourth = LinRange(-tp.amp, 0, round(Int, num_samples))

    pulse = repeat(vcat(first, second, third, fourth), tp.numpulses)
    fix_num = steps - length(pulse)
    pulse = vcat(pulse, zeros(Int, fix_num))
    return (; pulse, step = 1, pulseval = pulse[1])
end

function Processes.step!(::TrianglePulseA, context::C) where {C}
    (; pulse, step, hamiltonian) = context
    pulseval = pulse[step]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct BiasA{T} <: ProcessAlgorithm
    amp::T
end

function Processes.init(tp::BiasA, args)
    steps = num_calls(args)
    bias = ones(round(Int, steps)) .* tp.amp
    fix_num = steps - length(bias)
    bias = vcat(bias, zeros(Int, fix_num))
    return (; bias, step = 1, pulseval = bias[1])
end

function Processes.step!(::BiasA, context::C) where {C}
    (; bias, step, hamiltonian) = context
    pulseval = bias[step]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct SinPulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end

function Processes.init(tp::SinPulseA, args)
    steps = num_calls(args)
    theta = LinRange(0, 2pi * tp.numpulses, round(Int, steps))
    sins = tp.amp .* sin.(theta)
    return (; sins, step = 1, pulseval = sins[1])
end

function Processes.step!(::SinPulseA, context::C) where {C}
    (; sins, step, hamiltonian) = context
    pulseval = sins[step]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct LinAnealingA{T} <: ProcessAlgorithm
    start_T::T
    stop_T::T
end

function Processes.init(tp::LinAnealingA, args)
    n_calls = num_calls(args)
    dT = (tp.stop_T - tp.start_T) / n_calls
    return (; current_T = tp.start_T, dT)
end

function Processes.step!(::LinAnealingA, context::C) where {C}
    (; current_T, dT, model) = context
    temp!(model, max(current_T, 0))
    return (; current_T = current_T + dT)
end

struct LinAnealingB{T} <: ProcessAlgorithm
    start_T::T
    stop_T::T
end

function Processes.init(tp::LinAnealingB, args)
    steps = num_calls(args)
    num_samples = steps / 2
    first = LinRange(tp.start_T, tp.stop_T, round(Int, num_samples))
    second = LinRange(tp.stop_T, tp.start_T, round(Int, num_samples))
    tem_pulse = vcat(first, second)
    return (; tem_pulse, step = 1, temval = tem_pulse[1])
end

function Processes.step!(::LinAnealingB, context::C) where {C}
    (; tem_pulse, step, model) = context
    temval = tem_pulse[step]
    temp!(model, max(temval, 0))
    return (; step = step + 1, temval)
end

struct ValueLogger{Name} <: ProcessAlgorithm end
ValueLogger(name) = ValueLogger{Symbol(name)}()

function Processes.init(::ValueLogger, args)
    values = Float32[]
    processsizehint!(values, args)
    return (; values)
end

function Processes.step!(::ValueLogger, context::C) where {C}
    (; values, value) = context
    push!(values, value)
    return (;)
end

struct Recalc{I} <: Processes.ProcessAlgorithm end
Recalc(i) = Recalc{Int(i)}()

function Processes.step!(::Recalc{I}, context) where {I}
    (; hamiltonian) = context
    recalc!(hamiltonian[I])
    return (;)
end

struct ImageCapture{Name,F} <: ProcessAlgorithm
    min::F
    max::F
end

ImageCapture(name, min, max) = ImageCapture{Symbol(name), typeof(min)}(min, max)

function Processes.init(::ImageCapture, input)
    (; filepath) = input
    return (; callnum = 1, filepath)
end

function Processes.step!(ic::ImageCapture, context::C) where {C}
    (; array, filepath, callnum) = context
    if !(array isa AbstractArray{<:Real,3})
        @warn "ImageCapture expects a 3D numeric array" typeof(array)
        return (;)
    end

    CairoMakie.activate!()
    nx, ny, nz = size(array)
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
        cs[k] = array[x, y, z]
        k += 1
    end

    cmin, cmax = minmax(ic.min, ic.max)
    fig = Figure(size = (1000, 800))
    ax = Axis3(fig[1, 1]; xlabel = "x", ylabel = "y", zlabel = "z",
        aspect = (1, 1, 1), azimuth = 1.15, elevation = 0.35, title = "3D state")
    scatter!(ax, xs, ys, zs; color = cs, colormap = [:red, :black],
        colorrange = (cmin, cmax), markersize = 10)
    Colorbar(fig[1, 2]; colormap = [:red, :black], colorrange = (cmin, cmax), label = "value")

    mkpath(filepath)
    path = joinpath(filepath, "capture3d_$(callnum)_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".png")
    try
        save(path, fig)
    catch err
        @warn "Failed to save 3D capture image" err
    finally
        try
            close(fig)
        catch
        end
    end

    return (; callnum = callnum + 1)
end

struct DatatoDataframe{Name} <: ProcessAlgorithm end
DatatoDataframe(name) = DatatoDataframe{Symbol(name)}()

function Processes.init(::DatatoDataframe, input)
    (; filepath) = input
    return (; callnum = 1, filepath)
end

dimnames(i) = (:x, :y, :z)[i]

function Processes.step!(::DatatoDataframe, context::C) where {C}
    (; array, filepath, callnum) = context
    dimvecs = (;)
    for i in 1:ndims(array)
        dimvecs = (; dimvecs..., dimnames(i) => Int[])
    end
    df = DataFrame(; dimvecs..., value = eltype(array)[])
    mkpath(filepath)
    path = joinpath(filepath, "Df_running_$(callnum)_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".csv")
    try
        save(path, df)
    catch err
        @warn "Failed to save DataFrame" err
    end
    return (; callnum = callnum + 1)
end

function normalize_adj_by_average_col!(adj::A, scaling = one(eltype(adj))) where {A}
    sp = adj.sp
    cols = eltype(sp)[]
    for j in axes(sp, 2)
        push!(cols, sum(abs, @view sp[:, j]))
    end
    return sp .*= (scaling / mean(cols))
end
