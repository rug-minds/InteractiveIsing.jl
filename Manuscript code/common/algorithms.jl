struct TrianglePulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end

accepted_proposal_delta(proposal::InteractiveIsing.FlipProposal) = InteractiveIsing.accepteddelta(proposal)

function accepted_proposal_delta(proposal::InteractiveIsing.MultiSpinProposal)
    total = zero(eltype(proposal))
    @inbounds for i in 1:length(proposal)
        total += InteractiveIsing.accepteddelta(proposal, i)
    end
    return total
end

function StatefulAlgorithms.init(tp::TrianglePulseA, args)
    steps = num_calls(args)
    segments = 4 * tp.numpulses
    steps % segments == 0 || throw(ArgumentError(
        "TrianglePulseA requires num_calls to be divisible by 4 * numpulses; got num_calls=$steps and numpulses=$(tp.numpulses).",
    ))
    samples = div(steps, segments)
    segment(start, stop) = samples == 1 ? [start] : collect(LinRange(start, stop, samples))
    cycle = vcat(
        segment(zero(tp.amp), tp.amp),
        segment(tp.amp, zero(tp.amp)),
        segment(zero(tp.amp), -tp.amp),
        segment(-tp.amp, zero(tp.amp)),
    )
    pulse = repeat(cycle, tp.numpulses)
    return (; pulse, step = 1, pulseval = pulse[1])
end

function StatefulAlgorithms.step!(::TrianglePulseA, context::C) where {C}
    (; pulse, step, hamiltonian) = context
    pulseval = pulse[step]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct BiasA{T} <: ProcessAlgorithm
    amp::T
end

function StatefulAlgorithms.init(tp::BiasA, args)
    steps = num_calls(args)
    bias = ones(round(Int, steps)) .* tp.amp
    fix_num = steps - length(bias)
    bias = vcat(bias, zeros(Int, fix_num))
    return (; bias, step = 1, pulseval = bias[1])
end

function StatefulAlgorithms.step!(::BiasA, context::C) where {C}
    (; bias, step, hamiltonian) = context
    pulseval = bias[step]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct SinPulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end

function StatefulAlgorithms.init(tp::SinPulseA, args)
    steps = num_calls(args)
    theta = LinRange(0, 2pi * tp.numpulses, round(Int, steps))
    sins = tp.amp .* sin.(theta)
    return (; sins, step = 1, pulseval = sins[1])
end

function StatefulAlgorithms.step!(::SinPulseA, context::C) where {C}
    (; sins, step, hamiltonian) = context
    pulseval = sins[step]
    hamiltonian.b[] = pulseval
    return (; step = step + 1, pulseval)
end

struct LinAnealingA{T} <: ProcessAlgorithm
    start_T::T
    stop_T::T
end

function StatefulAlgorithms.init(tp::LinAnealingA, args)
    n_calls = num_calls(args)
    dT = (tp.stop_T - tp.start_T) / n_calls
    return (; current_T = tp.start_T, dT)
end

function StatefulAlgorithms.step!(::LinAnealingA, context::C) where {C}
    (; current_T, dT, model) = context
    temp!(model, max(current_T, 0))
    return (; current_T = current_T + dT)
end

struct LinAnealingB{T} <: ProcessAlgorithm
    start_T::T
    stop_T::T
end

function StatefulAlgorithms.init(tp::LinAnealingB, args)
    steps = num_calls(args)
    num_samples = steps / 2
    first = LinRange(tp.start_T, tp.stop_T, round(Int, num_samples))
    second = LinRange(tp.stop_T, tp.start_T, round(Int, num_samples))
    tem_pulse = vcat(first, second)
    return (; tem_pulse, step = 1, temval = tem_pulse[1])
end

function StatefulAlgorithms.step!(::LinAnealingB, context::C) where {C}
    (; tem_pulse, step, model) = context
    temval = tem_pulse[step]
    temp!(model, max(temval, 0))
    return (; step = step + 1, temval)
end

struct ValueLogger{Name} <: ProcessAlgorithm end
ValueLogger(name) = ValueLogger{Symbol(name)}()

function StatefulAlgorithms.init(::ValueLogger, args)
    values = Float32[]
    processsizehint!(values, args)
    return (; values)
end

function StatefulAlgorithms.step!(::ValueLogger, context::C) where {C}
    (; values, value) = context
    push!(values, value)
    return (;)
end

mean_polarization_zlayer(model, zidx::Integer) =
    mean(@view graph_array(model)[:, :, zidx])

mean_polarization_top(model) =
    mean_polarization_zlayer(model, size(graph_array(model), 3))

mean_polarization_mid(model) =
    mean_polarization_zlayer(model, cld(size(graph_array(model), 3), 2))

mean_polarization_bottom(model) =
    mean_polarization_zlayer(model, 1)

function staggered_z_polarization(model)
    A = graph_array(model)
    total = zero(Float64)
    for z in axes(A, 3)
        total += (isodd(z) ? 1 : -1) * sum(@view A[:, :, z])
    end
    return total / length(A)
end

function find_first_term(hamiltonian, ::Type{T}) where {T}
    for term in InteractiveIsing.hamiltonians(hamiltonian)
        term isa T && return term
    end
    return nothing
end

function sum_term_energy_by_type(hamiltonian, model, ::Type{T}) where {T}
    total = 0.0f0
    for term in InteractiveIsing.hamiltonians(hamiltonian)
        term isa T || continue
        total += Float32(InteractiveIsing.calculate(InteractiveIsing.H(), term, model))
    end
    return total
end

function bilinear_total_energy(hamiltonian, model)
    spins = graph_array(model)
    total = 0.0f0
    for term in InteractiveIsing.hamiltonians(hamiltonian)
        term isa InteractiveIsing.Bilinear || continue
        J = hasproperty(term.J, :sp) ? term.J.sp : term.J
        state_vector = vec(spins)
        total -= 0.5f0 * Float32(dot(state_vector, J * state_vector))
    end
    return total
end

function total_polynomial_energy(hamiltonian, model)
    spins = graph_array(model)
    total = 0.0f0
    for term in InteractiveIsing.hamiltonians(hamiltonian)
        term isa InteractiveIsing.PolynomialHamiltonian || continue
        for i in eachindex(spins)
            total += Float32(InteractiveIsing.calculate(InteractiveIsing.H_i(), term, model, i))
        end
    end
    return total
end

function coulomb_total_energy(hamiltonian)
    term = hamiltonian isa InteractiveIsing.CoulombHamiltonian ? hamiltonian :
        find_first_term(hamiltonian, InteractiveIsing.CoulombHamiltonian)
    isnothing(term) && return 0.0f0
    return 0.5f0 * Float32(sum(term.ρ .* term.u))
end

function total_supported_energy(hamiltonian, model)
    return bilinear_total_energy(hamiltonian, model) +
        sum_term_energy_by_type(hamiltonian, model, InteractiveIsing.MagField) +
        total_polynomial_energy(hamiltonian, model) +
        coulomb_total_energy(hamiltonian)
end

function coulomb_local_scale(model, term)
    values = Float64[]
    spins = graph_array(model)
    sizehint!(values, length(spins))
    for i in eachindex(spins)
        proposal = InteractiveIsing.SingleSpinProposal(i, spins[i], spins[i], 1)
        push!(values, abs(InteractiveIsing.calculate(
            InteractiveIsing.d_iH(),
            term,
            model,
            proposal,
        )))
    end
    return (; mean = mean(values), median = median(values), maximum = maximum(values))
end

struct DepolLogger{Name} <: ProcessAlgorithm end
DepolLogger(name) = DepolLogger{Symbol(name)}()

function StatefulAlgorithms.init(::DepolLogger, args)
    names = (
        :means, :medians, :maxima, :total_energy, :depol_energy,
        :interaction_energy, :field_energy, :poly_energy,
    )
    buffers = map(names) do _
        values = Float32[]
        processsizehint!(values, args)
        values
    end
    return NamedTuple{names}(buffers)
end

function StatefulAlgorithms.step!(::DepolLogger, context::C) where {C}
    (; means, medians, maxima, total_energy, depol_energy,
        interaction_energy, field_energy, poly_energy, model, hamiltonian) = context

    push!(total_energy, total_supported_energy(hamiltonian, model))
    push!(interaction_energy, bilinear_total_energy(hamiltonian, model))
    push!(field_energy, sum_term_energy_by_type(hamiltonian, model, InteractiveIsing.MagField))
    push!(poly_energy, total_polynomial_energy(hamiltonian, model))

    depol_term = find_first_term(hamiltonian, InteractiveIsing.CoulombHamiltonian)
    if isnothing(depol_term)
        push!(depol_energy, 0.0f0)
        push!(means, 0.0f0)
        push!(medians, 0.0f0)
        push!(maxima, 0.0f0)
    else
        depol = coulomb_local_scale(model, depol_term)
        push!(depol_energy, coulomb_total_energy(depol_term))
        push!(means, Float32(depol.mean))
        push!(medians, Float32(depol.median))
        push!(maxima, Float32(depol.maximum))
    end
    return (;)
end

struct Recalc{I} <: StatefulAlgorithms.ProcessAlgorithm end
Recalc(i) = Recalc{Int(i)}()

function StatefulAlgorithms.step!(::Recalc{I}, context) where {I}
    (; hamiltonian) = context
    recalc!(hamiltonian[I])
    return (;)
end

struct ImageCapture{Name,F} <: ProcessAlgorithm
    min::F
    max::F
end

ImageCapture(name, min, max) = ImageCapture{Symbol(name), typeof(min)}(min, max)

function StatefulAlgorithms.init(::ImageCapture, input)
    (; filepath) = input
    return (; callnum = 1, filepath)
end

function StatefulAlgorithms.step!(ic::ImageCapture, context::C) where {C}
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

function StatefulAlgorithms.init(::DatatoDataframe, input)
    (; filepath) = input
    return (; callnum = 1, filepath)
end

dimnames(i) = (:x, :y, :z)[i]

function StatefulAlgorithms.step!(::DatatoDataframe, context::C) where {C}
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
