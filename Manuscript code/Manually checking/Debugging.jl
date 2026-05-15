using InteractiveIsing, GLMakie, FileIO, CairoMakie
using InteractiveIsing.Processes
import InteractiveIsing as II

using Dates
using DataFrames
using XLSX
using Random
using StatsBase

## Utility functions for experiments
### Use ii. to check if the terms are correct
### Now the H is written like H_self + H_quartic
### Which is Jii*Si^2 + Qc*Jii*Si^4 wichi means Jii=a, Qc*Jii=b in a*Si^2 + b*Si^4



function newmakie(makietype, args...; show = show_figures, kwargs...)
    f = makietype(args...; kwargs...)
    if show
        scr = GLMakie.Screen()
        display(scr, f)
    end
    f
end

function makieaxis(axisfunc, modifiers...; show = show_figures, savepath = nothing)
    f = Figure()
    ax = axisfunc(f[1, 1])
    for mod in modifiers
        mod(ax)
    end
    if show
        scr = GLMakie.Screen()
        display(scr, f)
    end
    if !isnothing(savepath)
        save(savepath, f)
    end
    f
end

accepted_proposal_delta_base(proposal::II.FlipProposal) = II.accepteddelta(proposal)

function accepted_proposal_delta_base(proposal::II.MultiSpinProposal)
    total = zero(eltype(proposal))
    @inbounds for i in 1:length(proposal)
        total += II.accepteddelta(proposal, i)
    end
    return total
end

function select_dynamics(g, algorithm_name::Symbol; algorithm_kwargs = (;))
    algorithm_name in (:default, :metropolis) && return g.default_algorithm
    algorithm_name == :local_langevin && return LocalLangevin(; algorithm_kwargs...)
    algorithm_name == :global_langevin && return GlobalLangevin(; algorithm_kwargs...)
    algorithm_name == :block_langevin && return BlockLangevin(; algorithm_kwargs...)

    error("Unknown algorithm_name $(repr(algorithm_name)). Use :default, :metropolis, :local_langevin, :global_langevin, or :block_langevin.")
end

function mean_polarization_zlayer(model, zidx::Integer)
    return Float32(mean(state(model[1])[:, :, zidx]))
end

function staggered_z_polarization(model)
    s = state(model[1])
    total = 0.0f0
    _, _, zL = size(s)
    @inbounds for z in 1:zL
        total += (isodd(z) ? 1.0f0 : -1.0f0) * Float32(mean(s[:, :, z]))
    end
    return total / zL
end


# Preferred naming:
# - `wf_*` = weight-function definitions
# - use `dr` when only scalar physical distance matters
# - use `dc` when the direction / shell / parity matters
# `dr` is already built from `LatticeConstants(...)` inside the adjacency builder.


function wf_ferro_r(J, p; dr)
    return J / dr^p
end

function wf_ferro_inv_r2(; dr)
    return wf_ferro_r(1.0, 2; dr)
end

function wf_ferro_NN1(J; dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    s == 1 || return 0.0
    return J
end

function wf_antiferro3D_r(J, p; dr)
    return -J / dr^p
end

function wf_antiferro3D_NN1(J; dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    s == 1 || return 0.0
    return -J
end

function wf_antiferroX_r(J; dr, dc)
    dx, dy, dz = dc
    prefac = if dx == 0 && dy == 0
        J
    elseif dx == 0
        J
    else
        -J
    end
    return prefac / dr
end

function wf_antiferroXY_r(J; dr, dc)
    dx, dy, dz = dc
    prefac = if dx == 0 && dy == 0
        J
    else
        -J
    end
    return prefac / dr
end

function wf_antiferroDiag_r(J; dr, dc)
    dx, dy, dz = dc
    prefac = if dz == 0 && abs(dx) == abs(dy) && dx != 0
        -J
    else
        J
    end
    return prefac / dr
end

function wf_shell_ferro(c_short, lambda_2, lambda_3; dc)
    dx, dy, dz = dc
    k1 = 1.0
    k2 = lambda_2 * k1
    k3 = lambda_3 * k2
    s = dx*dx + dy*dy + dz*dz

    prefac_sr = if s == 1
        k1
    elseif s == 2
        k2
    elseif s == 3
        k3
    else
        0.0
    end

    return c_short * prefac_sr
end

function wf_xy_checkerboard_antiferro_inv_r2(; dc)
    dx, dy, dz = dc
    if (abs(dx) + abs(dy)) % 2 == 0
        return 1.0 / norm2(dc)
    else
        return -1.0 / norm2(dc)
    end
end

function wf_skyrmion_like(; dc)
    dx, dy, dz = dc
    prefac = (abs(dx) > 0 || abs(dy) > 0) ? -2.0 : 2.0
    return prefac / norm2(dc)
end

function wf_nn_anisotropic(Jz, ratio_xy; dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    s == 1 || return 0.0
    return dz != 0 ? Jz : ratio_xy * Jz
end

function wf_nn_anisotropic_inv_r(Jz, ratio_xy; dr, dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    s == 1 || return 0.0
    prefac = dz != 0 ? Jz : ratio_xy * Jz
    return prefac / dr
end

function wf_layered_afe_nn(Jz, ratio_xy; dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    s == 1 || return 0.0
    return dz != 0 ? -Jz : ratio_xy * Jz
end

function wf_shell_competing(c_short, lambda_2, lambda_3; dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    if s == 1
        return c_short
    elseif s == 2
        return -lambda_2 * c_short
    elseif s == 3
        return lambda_3 * c_short
    else
        return 0.0
    end
end

function wf_plane_checkerboard_nn(Jz; dc)
    dx, dy, dz = dc
    s = dx*dx + dy*dy + dz*dz
    s == 1 || return 0.0
    if dz != 0
        return Jz
    elseif (abs(dx) + abs(dy)) == 1
        return -Jz
    else
        return 0.0
    end
end

##################################################################################
### struct start: TrianglePulseA (simple four-segment triangular waveform)
### Run with TrianlePulseA
###  /\
### /  \    
###     \  /
###      \/

struct TrianglePulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end
function Processes.init(tp::TrianglePulseA, args)
    amp = tp.amp
    numpulses = tp.numpulses
    steps = num_calls(args)
    num_samples = steps/(4*numpulses)
    first  = LinRange(0, amp, round(Int,num_samples))
    second = LinRange(amp, 0, round(Int,num_samples))
    third  = LinRange(0, -amp, round(Int,num_samples))
    fourth = LinRange(-amp, 0, round(Int,num_samples))

    pulse = vcat(first, second, third, fourth)
    pulse = repeat(pulse, numpulses)

    fix_num = num_calls(args) - length(pulse)
    fix_arr = zeros(Int, fix_num)
    pulse   = vcat(pulse, fix_arr)

    # Predefine storage arrays
    step = 1
    return (;pulse, step, pulseval = pulse[1])
end
function Processes.step!(::TrianglePulseA, context::C) where C
    (;pulse, step, hamiltonian) = context
    pulseval = pulse[step]
    hamiltonian.b[] = pulseval

    return (;step = step + 1, pulseval)
end
### struct end: TrianglePulseA
##################################################################################

##################################################################################
### struct start: Snapshot (simple four-segment triangular waveform)
struct Snapshot{DataType, Name} <: ProcessAlgorithm end

function Processes.step!(::Snapshot{DT, Name}, context::C) where {DT, Name, C}
    (;data) = context
    saveimg(data)
end

function saveimg(g::IsingGraph)
    ######
end
### struct end: Snapshot
##################################################################################


##################################################################################
### struct start: BiasA (stable bias)
### Run with BiasA 
### 
### _____
###      

struct BiasA{T} <: ProcessAlgorithm
    amp::T
end
function Processes.init(tp::BiasA, args)
    amp = tp.amp
    steps = num_calls(args)
    bias  = ones(round(Int, steps)) .* amp

    fix_num = num_calls(args) - length(bias)
    fix_arr = zeros(Int, fix_num)
    bias   = vcat(bias, fix_arr)

    # Predefine storage arrays
    step = 1
    return (;bias, step, pulseval = bias[1])
end
function Processes.step!(::BiasA, context::C) where C
    (;bias, step, hamiltonian) = context
    pulseval = bias[step]
    hamiltonian.b[] = pulseval
    return (;step = step + 1, pulseval)
end
### struct end: BiasA
##################################################################################


##################################################################################
### struct start: SinPulseA (simple sine waveform)
### Run with SinPulseA

struct SinPulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end    
function Processes.init(tp::SinPulseA, args)
    amp = tp.amp
    numpulses = tp.numpulses
    steps = num_calls(args)
    max_theta = 2*pi * numpulses

    theta = LinRange(0, max_theta, round(Int,steps))
    sins = amp .* sin.(theta)
    step = 1
    return (;sins, step, pulseval = sins[1])
end
function Processes.step!(::SinPulseA, context::C) where C
    (;sins, step, hamiltonian) = context
    pulse_val = sins[step]
    hamiltonian.b[] = pulse_val
    return (;step = step + 1, pulseval = pulse_val)
end
### struct end: SinPulseA
##################################################################################



##################################################################################
### struct start: TemAnealingA (simple sine waveform)
### Run with TemAnealingA
struct LinAnealingA{T} <: ProcessAlgorithm
    start_T::T
    stop_T::T
end  
function Processes.init(tp::LinAnealingA, args)
    n_calls = num_calls(args)
    dT = (tp.stop_T - tp.start_T) / n_calls
    (;current_T = tp.start_T, dT)
end
function Processes.step!(::LinAnealingA, context::C) where C
    (;current_T, dT, model) = context
    temp!(model, max(current_T, 0))
    return (;current_T = current_T + dT)
end
##################################################################################

##################################################################################
### struct start: LinAnealingB (simple sine waveform)
### Run with LinAnealingB
struct LinAnealingB{T} <: ProcessAlgorithm
    start_T::T
    stop_T::T
end
LinAnealingB(start_T, stop_T) = LinAnealingB(promote(start_T, stop_T)...)
function Processes.init(tp::LinAnealingB, args)
    start_T = tp.start_T
    stop_T = tp.stop_T
    steps = num_calls(args)
    num_samples = steps/2
    first  = LinRange(start_T, stop_T, round(Int,num_samples))
    second = LinRange(stop_T, start_T, round(Int,num_samples))

    tem_pulse = vcat(first, second)

    # Predefine storage arrays
    step = 1
    return (;tem_pulse, step, temval = tem_pulse[1])
end
function Processes.step!(::LinAnealingB, context::C) where C
    (;tem_pulse, step, model) = context

    temval = tem_pulse[step]

    temp!(model, max(temval, 0))

    return (;step = step + 1, temval)
end
##################################################################################


##################################################################################
struct ValueLogger{Name} <: ProcessAlgorithm end
ValueLogger(name) = ValueLogger{Symbol(name)}()
function Processes.init(::ValueLogger, args)
    values = Float32[]
    processsizehint!(values, args)
    (;values)
end
function Processes.step!(::ValueLogger, context::C) where C
    (;values, value) = context
    push!(values, value)
    return (;)
end
##################################################################################

##################################################################################
struct DepolLogger{Name} <: ProcessAlgorithm end
DepolLogger(name) = DepolLogger{Symbol(name)}()
function Processes.init(::DepolLogger, args)
    means = Float32[]
    medians = Float32[]
    maxima = Float32[]
    total_energy = Float32[]
    depol_energy = Float32[]
    interaction_energy = Float32[]
    field_energy = Float32[]
    poly_energy = Float32[]
    processsizehint!(means, args)
    processsizehint!(medians, args)
    processsizehint!(maxima, args)
    processsizehint!(total_energy, args)
    processsizehint!(depol_energy, args)
    processsizehint!(interaction_energy, args)
    processsizehint!(field_energy, args)
    processsizehint!(poly_energy, args)
    (; means, medians, maxima, total_energy, depol_energy, interaction_energy, field_energy, poly_energy)
end
function Processes.step!(::DepolLogger, context::C) where C
    (; means, medians, maxima, total_energy, depol_energy, interaction_energy, field_energy, poly_energy, model, hamiltonian) = context
    depol_term = find_first_term(hamiltonian, InteractiveIsing.CoulombHamiltonian)
    push!(total_energy, total_supported_energy(hamiltonian, model))
    push!(interaction_energy, bilinear_total_energy(hamiltonian, model))
    push!(field_energy, sum_term_energy_by_type(hamiltonian, model, InteractiveIsing.MagField))
    push!(poly_energy, total_polynomial_energy(hamiltonian, model))
    isnothing(depol_term) && return (;)
    depol = coulomb_local_scale(model, depol_term)
    push!(depol_energy, coulomb_total_energy(depol_term))
    push!(means, Float32(depol.mean))
    push!(medians, Float32(depol.median))
    push!(maxima, Float32(depol.maximum))
    return (;)
end
##################################################################################

##################################################################################
struct Recalc{I} <: Processes.ProcessAlgorithm end
Recalc(i) = Recalc{Int(i)}()
function Processes.step!(r::Recalc{I}, context) where I
    (;hamiltonian) = context
    recalc!(hamiltonian[I])
    return (;)
end
##################################################################################


##################################################################################
struct ImageCapture{Name,F} <: ProcessAlgorithm
    min::F
    max::F
    # filepath::Symbol
end
# fix: store (min,max) in the right order
# ImageCapture(name, min, max; filepath = pwd()) = ImageCapture{Symbol(name), typeof(min)}(min, max, Symbol(filepath))
ImageCapture(name, min, max) = ImageCapture{Symbol(name), typeof(min)}(min, max)
function Processes.init(ic::ImageCapture, input)
    (;filepath) = input
    (;callnum = 1, filepath)
end
function Processes.step!(ic::ImageCapture, context::C) where C
    (;array, filepath, callnum) = context
    A = array
    if !(A isa AbstractArray{<:Real,3})
        @warn "ImageCapture expects a 3D numeric array" typeof(A)
        return (;)
    end
    CairoMakie.activate!()
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

    cmin = ic.min
    cmax = ic.max
    if cmin > cmax
        cmin, cmax = cmax, cmin
    end

    # use a simple explicit colormap (avoid relying on cgrad availability)
    cmap = [:red, :black]

    fig = Figure(size = (1000, 800))
    ax = Axis3(
        fig[1, 1];
        xlabel = "x", ylabel = "y", zlabel = "z",
        aspect = (1, 1, 1),
        azimuth = 1.15,
        elevation = 0.35,
        title = "3D state"
    )
    scatter!(ax, xs, ys, zs;
        color = cs,
        colormap = cmap,
        colorrange = (cmin, cmax),
        markersize = 10
    )
    Colorbar(fig[1, 2]; colormap = cmap, colorrange = (cmin, cmax), label = "value")
    # outdir = ic.filepath |> string
    outdir = filepath
    mkpath(outdir)
    path = joinpath(outdir, "capture3d_$(callnum)_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".png")
    try
        save(path, fig)
    catch err
        @warn "Failed to save 3D capture image" err
    finally
        # avoid accumulating figures in a long-running process
        try
            close(fig)
        catch
        end
    end

    return (;callnum = callnum + 1)
end
##################################################################################

##################################################################################
struct DatatoDataframe{Name} <: ProcessAlgorithm
end
# fix: store (min,max) in the right order
# ImageCapture(name, min, max; filepath = pwd()) = ImageCapture{Symbol(name), typeof(min)}(min, max, Symbol(filepath))
DatatoDataframe(name) = DatatoDataframe{Symbol(name)}()
function Processes.init(ic::DatatoDataframe, input)
    (;filepath) = input
    (;callnum = 1, filepath)
end
dimnames(i) = (:x, :y, :z)[i]
function Processes.step!(ic::DatatoDataframe, context::C) where C
    (;array, filepath, callnum) = context
    A = array
    arraysize = size(A)
    dimvecs = (;)
    for i in 1:length(arraysize)
        dimvecs = (;dimvecs..., dimnames(i) => Int[])
    end
    df = DataFrame(;dimvecs..., value = eltype(A)[])
    outdir = filepath
    path = joinpath(outdir, "Df_running_$(callnum)_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".csv")
    try
        save(path, df)
    catch err
        @warn "Failed to save DataFrame" err
    finally
        # avoid accumulating figures in a long-running process
        try
            close(fig)
        catch
        end
    end
    return (;callnum = callnum + 1)
end
##################################################################################
function normalize_adj_by_average_col!(adj::A, scaling = one(eltype(adj))) where A
    adj = adj.sp
    cols = eltype(adj)[]
    for j in axes(adj, 2)
        s = sum(abs, @view adj[:, j])
        push!(cols, s)
    end
    avg_col_sum = mean(cols)
    return adj .*= (scaling/avg_col_sum) 
end
function IntegrateAndLog(type = Float64, loginterval = 1)
    integrator = Integrator(type, name = :integrate_and_log)
    logger = Logger(type, name = :integrate_and_log)
    c = @CompositeAlgorithm begin
        @alias integrator = integrator
        @alias logger = logger

        total = integrator()
        @every loginterval logger(value = @transform(x -> x[], total))
    end
    pack = package(c)
end
#####################################################################################
#####################################################################################

function landau_energy(P, a, b, c, d, e)
    return a * P^2 + b * P^4 + c * P^6 + d * P^8 + e * P^10
end

function estimate_landau_barrier(a, b, c, d, e; Pmin = -1.5, Pmax = 1.5, ngrid = 20001)
    xs = range(Pmin, Pmax, length = ngrid)
    ys = [landau_energy(x, a, b, c, d, e) for x in xs]

    min_idxs = Int[]
    max_idxs = Int[]
    for i in 2:(length(xs) - 1)
        yi = ys[i]
        if yi <= ys[i - 1] && yi <= ys[i + 1]
            push!(min_idxs, i)
        end
        if yi >= ys[i - 1] && yi >= ys[i + 1]
            push!(max_idxs, i)
        end
    end

    pos_min_idxs = filter(i -> xs[i] > 0, min_idxs)
    isempty(pos_min_idxs) && error("No positive local minimum found in the scanned Landau window.")
    well_idx = pos_min_idxs[argmin(ys[pos_min_idxs])]
    P0 = xs[well_idx]
    Ewell = ys[well_idx]

    between_max_idxs = filter(i -> 0 <= xs[i] <= P0, max_idxs)
    barrier_idx = isempty(between_max_idxs) ? argmin(abs.(xs)) : between_max_idxs[argmax(ys[between_max_idxs])]
    Ps = xs[barrier_idx]
    Ebarrier = ys[barrier_idx]

    return (; P0, Ps, ΔF = Ebarrier - Ewell, Ewell, Ebarrier)
end

function avg_interaction_scale(adj_like)
    A = hasproperty(adj_like, :sp) ? adj_like.sp : adj_like
    colsums = Float64[]
    for j in axes(A, 2)
        push!(colsums, sum(abs, @view A[:, j]))
    end
    return (; SJ = mean(colsums), SJ_min = minimum(colsums), SJ_max = maximum(colsums))
end

function find_first_term(hts, ::Type{T}) where T
    for h in InteractiveIsing.hamiltonians(hts)
        h isa T && return h
    end
    return nothing
end

function sum_term_energy_by_type(hamiltonian, model, ::Type{T}) where T
    total = 0.0f0
    for h in InteractiveIsing.hamiltonians(hamiltonian)
        h isa T && (total += Float32(InteractiveIsing.calculate(InteractiveIsing.H(), h, model)))
    end
    return total
end

function bilinear_total_energy(hamiltonian, model)
    s = state(model)
    total = 0.0f0
    for h in InteractiveIsing.hamiltonians(hamiltonian)
        if h isa InteractiveIsing.Bilinear
            local_total = 0.0f0
            for i in eachindex(s)
                local_total += Float32(s[i]) * Float32(InteractiveIsing.calculate(InteractiveIsing.d_iH(), h, model, i))
            end
            total += 0.5f0 * local_total
        end
    end
    return total
end

function total_polynomial_energy(hamiltonian, model)
    total = 0.0f0
    s = state(model)
    for h in InteractiveIsing.hamiltonians(hamiltonian)
        if h isa InteractiveIsing.PolynomialHamiltonian
            local_total = 0.0f0
            for i in eachindex(s)
                local_total += Float32(InteractiveIsing.calculate(InteractiveIsing.H_i(), h, model, i))
            end
            total += local_total
        end
    end
    return total
end

function coulomb_total_energy(hamiltonian)
    depol_term = hamiltonian isa InteractiveIsing.CoulombHamiltonian ? hamiltonian :
        find_first_term(hamiltonian, InteractiveIsing.CoulombHamiltonian)
    isnothing(depol_term) && return 0.0f0
    return 0.5f0 * Float32(sum(depol_term.ρ .* depol_term.u))
end

function total_supported_energy(hamiltonian, model)
    total = 0.0f0
    total += bilinear_total_energy(hamiltonian, model)
    total += sum_term_energy_by_type(hamiltonian, model, InteractiveIsing.MagField)
    total += total_polynomial_energy(hamiltonian, model)
    total += coulomb_total_energy(hamiltonian)
    return total
end

function coulomb_local_scale(g, coulomb_term)
    vals = Float64[]
    for i in eachindex(state(g))
        push!(vals, abs(InteractiveIsing.calculate(InteractiveIsing.d_iH(), coulomb_term, g, i)))
    end
    return (; mean = mean(vals), median = median(vals), maximum = maximum(vals))
end

function print_reduced_parameter_summary(; a, b, c, d, e, g, JIsing, Scale, Screening, field_typ = 1.0, defect_typ = 0.0, Pmin = -1.5, Pmax = 1.5)
    barrier = estimate_landau_barrier(a, b, c, d, e; Pmin, Pmax)
    interaction = avg_interaction_scale(adj(g))
    coulomb_term = find_first_term(g.hamiltonian, InteractiveIsing.CoulombHamiltonian)
    depol = isnothing(coulomb_term) ? nothing : coulomb_local_scale(g, coulomb_term)

    P0 = barrier.P0
    ΔF = barrier.ΔF
    SJ = interaction.SJ

    println()
    println("=== Reduced Parameter Summary ===")
    println("Input parameters")
    println("  JIsing               = ", JIsing)
    println("  Scale                = ", Scale)
    println("  Screening            = ", Screening)
    println("  field_typ            = ", field_typ)
    println("  defect_typ           = ", defect_typ)
    println()
    println("Directly computed reduced quantities")
    println("  P0                   = ", P0)
    println("  Ps                   = ", barrier.Ps)
    println("  DeltaF_barrier       = ", ΔF)
    println("  S_J                  = ", SJ)
    println("  S_J range            = [", interaction.SJ_min, ", ", interaction.SJ_max, "]")
    println("  Lambda_int           = ", (P0^2 * SJ) / ΔF)
    println("  Lambda_barrier       = ", ΔF / (P0^2 * SJ))
    println("  Lambda_field         = ", abs(field_typ) / (P0 * SJ))
    println("  Lambda_defect        = ", abs(defect_typ) / (P0 * SJ))
    println("  Theta_field          = ", abs(P0 * field_typ) / ΔF)
    println("  Theta_defect         = ", abs(P0 * defect_typ) / ΔF)

    if !isnothing(depol)
        println()
        println("State-dependent depolarization quantities")
        println("  E_dep_mean           = ", depol.mean)
        println("  E_dep_median         = ", depol.median)
        println("  E_dep_max            = ", depol.maximum, "  # extreme bound, not the main typical value")
        println("  Lambda_dep_mean      = ", depol.mean / (P0^2 * SJ))
        println("  Lambda_dep_median    = ", depol.median / (P0^2 * SJ))
        println("  Theta_dep_mean       = ", depol.mean / ΔF)
        println("  Theta_dep_median     = ", depol.median / ΔF)
    end
    println("=================================")
    println()
end

function with_saved_state(f, g, depol_term)
    saved_state = copy(state(g))
    saved_tracker = depol_term.recalc_tracker[]
    try
        return f()
    finally
        state(g) .= saved_state
        init!(depol_term, g)
        depol_term.recalc_tracker[] = saved_tracker
    end
end

function reference_coulomb_scale(g, depol_term, P0)
    return with_saved_state(g, depol_term) do
        state(g) .= P0
        init!(depol_term, g)
        vals = Float64[]
        for i in eachindex(state(g))
            local_field = abs(InteractiveIsing.calculate(InteractiveIsing.d_iH(), depol_term, g, i))
            push!(vals, 2 * abs(P0) * local_field)
        end
        (; mean = mean(vals), median = median(vals), maximum = maximum(vals))
    end
end

function print_reduced_parameter_summary_reference(; a, b, c, d, e, g, JIsing, Scale, Screening, field_typ = 1.0, defect_typ = 0.0, Pmin = -1.5, Pmax = 1.5)
    barrier = estimate_landau_barrier(a, b, c, d, e; Pmin, Pmax)
    interaction = avg_interaction_scale(adj(g))
    depol_term = find_first_term(g.hamiltonian, InteractiveIsing.CoulombHamiltonian)

    P0 = barrier.P0
    ΔF = barrier.ΔF
    SJ = interaction.SJ
    depol_ref = isnothing(depol_term) ? nothing : reference_coulomb_scale(g, depol_term, P0)
    depol_current = isnothing(depol_term) ? nothing : coulomb_local_scale(g, depol_term)
    rows = NamedTuple[]
    addrow!(section, key, value; note = "") = push!(rows, (; section, key, value, note))

    addrow!("input", "JIsing", JIsing)
    addrow!("input", "Scale", Scale)
    addrow!("input", "Screening", Screening)
    addrow!("input", "field_typ", field_typ)
    addrow!("input", "defect_typ", defect_typ)
    addrow!("direct", "P0", P0)
    addrow!("direct", "Ps", barrier.Ps)
    addrow!("direct", "DeltaF_barrier", ΔF)
    addrow!("direct", "S_J", SJ)
    addrow!("direct", "S_J_min", interaction.SJ_min)
    addrow!("direct", "S_J_max", interaction.SJ_max)
    addrow!("direct", "Lambda_int", (P0^2 * SJ) / ΔF)
    addrow!("direct", "Lambda_barrier", ΔF / (P0^2 * SJ))
    addrow!("direct", "Lambda_field", abs(field_typ) / (P0 * SJ))
    addrow!("direct", "Lambda_defect", abs(defect_typ) / (P0 * SJ))
    addrow!("direct", "Theta_field", abs(P0 * field_typ) / ΔF)
    addrow!("direct", "Theta_defect", abs(P0 * defect_typ) / ΔF)

    if !isnothing(depol_ref)
        addrow!("reference_depol", "reference_state", "all dipoles at +P0")
        addrow!("reference_depol", "E_dep_ref_mean", depol_ref.mean; note = "estimated full-flip depolarization work scale")
        addrow!("reference_depol", "E_dep_ref_median", depol_ref.median)
        addrow!("reference_depol", "E_dep_ref_max", depol_ref.maximum)
        addrow!("reference_depol", "Lambda_dep_ref_mean", depol_ref.mean / (P0^2 * SJ))
        addrow!("reference_depol", "Lambda_dep_ref_med", depol_ref.median / (P0^2 * SJ))
        addrow!("reference_depol", "Theta_dep_ref_mean", depol_ref.mean / ΔF)
        addrow!("reference_depol", "Theta_dep_ref_med", depol_ref.median / ΔF)
        addrow!("current_depol", "dH_dep_cur_mean", depol_current.mean; note = "state-dependent local derivative diagnostic")
        addrow!("current_depol", "dH_dep_cur_median", depol_current.median)
        addrow!("current_depol", "dH_dep_cur_max", depol_current.maximum)
    end

    println()
    println("=== Reduced Parameter Summary (Reference-State Depol) ===")
    println("Input parameters")
    println("  JIsing               = ", JIsing)
    println("  Scale                = ", Scale)
    println("  Screening            = ", Screening)
    println("  field_typ            = ", field_typ)
    println("  defect_typ           = ", defect_typ)
    println()
    println("Directly computed reduced quantities")
    println("  P0                   = ", P0)
    println("  Ps                   = ", barrier.Ps)
    println("  DeltaF_barrier       = ", ΔF)
    println("  S_J                  = ", SJ)
    println("  S_J range            = [", interaction.SJ_min, ", ", interaction.SJ_max, "]")
    println("  Lambda_int           = ", (P0^2 * SJ) / ΔF)
    println("  Lambda_barrier       = ", ΔF / (P0^2 * SJ))
    println("  Lambda_field         = ", abs(field_typ) / (P0 * SJ))
    println("  Lambda_defect        = ", abs(defect_typ) / (P0 * SJ))
    println("  Theta_field          = ", abs(P0 * field_typ) / ΔF)
    println("  Theta_defect         = ", abs(P0 * defect_typ) / ΔF)

    if !isnothing(depol_ref)
        println()
        println("Reference-state depolarization quantities")
        println("  Reference state      = all dipoles at +P0")
        println("  E_dep_ref_mean       = ", depol_ref.mean, "  # estimated full-flip depolarization work scale")
        println("  E_dep_ref_median     = ", depol_ref.median)
        println("  E_dep_ref_max        = ", depol_ref.maximum)
        println("  Lambda_dep_ref_mean  = ", depol_ref.mean / (P0^2 * SJ))
        println("  Lambda_dep_ref_med   = ", depol_ref.median / (P0^2 * SJ))
        println("  Theta_dep_ref_mean   = ", depol_ref.mean / ΔF)
        println("  Theta_dep_ref_med    = ", depol_ref.median / ΔF)
        println()
        println("Current-state depolarization diagnostics")
        println("  dH_dep_cur_mean      = ", depol_current.mean, "  # state-dependent local derivative diagnostic")
        println("  dH_dep_cur_median    = ", depol_current.median)
        println("  dH_dep_cur_max       = ", depol_current.maximum)
    end
    println("========================================================")
    println()
    return DataFrame(rows)
end

maybe_global(name::Symbol) = isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : nothing

function active_route_name()
    has_anneal = !isnothing(maybe_global(:Pr1))
    has_pulse = !isnothing(maybe_global(:Pr2))
    if has_anneal && has_pulse
        return "anneal_pulse"
    elseif has_anneal
        return "anneal"
    elseif has_pulse
        return "pulse"
    else
        return "run"
    end
end

function make_run_basename(; script_name = "Basefile", route = active_route_name(), timestamp = Dates.now())
    date_str = Dates.format(timestamp, "yyyymmdd_HHMMSS")
    short_id = randstring('A':'Z', 4) * randstring('0':'9', 2)
    return string(script_name, "_", route, "_", date_str, "_", short_id)
end

function numeric_or_missing_column(values, nrows)
    col = Vector{Union{Missing, Float64}}(missing, nrows)
    isnothing(values) && return col
    n = min(length(values), nrows)
    n == 0 && return col
    col[1:n] .= Float64.(values[1:n])
    return col
end

function wide_series_dataframe(columns::Pair{Symbol, <:Any}...)
    valid = filter(collect(columns)) do pair
        values = pair.second
        !isnothing(values) && hasmethod(length, Tuple{typeof(values)}) && length(values) > 0
    end
    isempty(valid) && return DataFrame()
    nrows = maximum(length(pair.second) for pair in valid)
    data = (; (pair.first => numeric_or_missing_column(pair.second, nrows) for pair in valid)...)
    return DataFrame(data)
end

function params_dataframe(params::Pair{String, <:Any}...)
    keys = String[first(pair) for pair in params]
    values = Any[last(pair) for pair in params]
    return DataFrame(key = keys, value = values)
end

function coefficient_summary_dataframe(coeffs::Pair{String, <:Any}...)
    rows = NamedTuple[]
    for (name, values) in coeffs
        isnothing(values) && continue
        v = Float64.(vec(values))
        push!(rows, (;
            coefficient = name,
            mean = mean(v),
            std = length(v) > 1 ? std(v) : 0.0,
            minimum = minimum(v),
            maximum = maximum(v),
            first_value = first(v),
            n = length(v),
        ))
    end
    return DataFrame(rows)
end

function collect_run_series_dataframe()
    return wide_series_dataframe(
        :anneal_voltage => maybe_global(:voltage1),
        :anneal_temperature => maybe_global(:Temp1),
        :anneal_Pr => maybe_global(:Pr1),
        :anneal_depol_mean => maybe_global(:depol_mean1),
        :anneal_depol_median => maybe_global(:depol_median1),
        :anneal_depol_max => maybe_global(:depol_max1),
        :anneal_H_total => maybe_global(:Htotal1),
        :anneal_H_dep => maybe_global(:Hdep1),
        :anneal_H_J => maybe_global(:HJ1),
        :anneal_H_field => maybe_global(:Hfield1),
        :anneal_H_poly => maybe_global(:Hpoly1),
        :anneal_H_rest => maybe_global(:Hrest1),
        :anneal_P_AFE_z => maybe_global(:PAFEz1),
        :anneal_P_top => maybe_global(:Ptop1),
        :anneal_P_mid => maybe_global(:Pmid1),
        :anneal_P_bot => maybe_global(:Pbot1),
        :pulse_voltage => maybe_global(:voltage2),
        :pulse_Pr => maybe_global(:Pr2),
        :pulse_depol_mean => maybe_global(:depol_mean2),
        :pulse_depol_median => maybe_global(:depol_median2),
        :pulse_depol_max => maybe_global(:depol_max2),
        :pulse_H_total => maybe_global(:Htotal2),
        :pulse_H_dep => maybe_global(:Hdep2),
        :pulse_H_J => maybe_global(:HJ2),
        :pulse_H_field => maybe_global(:Hfield2),
        :pulse_H_poly => maybe_global(:Hpoly2),
        :pulse_H_rest => maybe_global(:Hrest2),
        :pulse_P_AFE_z => maybe_global(:PAFEz2),
        :pulse_P_top => maybe_global(:Ptop2),
        :pulse_P_mid => maybe_global(:Pmid2),
        :pulse_P_bot => maybe_global(:Pbot2),
    )
end

function collect_figure_specs()
    return [
        "anneal_T_Pr" => maybe_global(:fTPr),
        "anneal_Pr_step" => maybe_global(:fPr_anneal),
        "anneal_order_step" => maybe_global(:fOrderStep_anneal),
        "anneal_order_temperature" => maybe_global(:fOrderT_anneal),
        "anneal_Hrest_Pr" => maybe_global(:fHrestPr_anneal),
        "anneal_Hrest_temperature" => maybe_global(:fHrestT_anneal),
        "anneal_Htotal_Pr" => maybe_global(:fHtotP_anneal),
        "anneal_Htotal_temperature" => maybe_global(:fHtotT_anneal),
        "anneal_Hterms_Pr" => maybe_global(:fHtermsPr_anneal),
        "anneal_Hterms_temperature" => maybe_global(:fHtermsT_anneal),
        "anneal_depol_Pr" => maybe_global(:fDepP_anneal),
        "anneal_depol_temperature" => maybe_global(:fDepT_anneal),
        "pulse_V_Pr" => maybe_global(:fVPr),
        "pulse_Pr_step" => maybe_global(:fPr),
        "pulse_order_step" => maybe_global(:fOrderStep),
        "pulse_order_voltage" => maybe_global(:fOrderV),
        "pulse_Hrest_Pr" => maybe_global(:fHrestPr),
        "pulse_Hrest_voltage" => maybe_global(:fHrestV),
        "pulse_Htotal_Pr" => maybe_global(:fHtotP),
        "pulse_Htotal_voltage" => maybe_global(:fHtotV),
        "pulse_Hterms_Pr" => maybe_global(:fHtermsPr),
        "pulse_Hterms_voltage" => maybe_global(:fHtermsV),
        "pulse_depol_Pr" => maybe_global(:fDepP),
        "pulse_depol_voltage" => maybe_global(:fDepV),
    ]
end

function save_existing_figures(outdir, base_name; enabled = true)
    enabled || return String[]
    saved_paths = String[]
    for (name, fig) in collect_figure_specs()
        isnothing(fig) && continue
        path = joinpath(outdir, base_name * "_" * name * ".png")
        try
            save(path, fig)
            push!(saved_paths, path)
        catch err
            @warn "Failed to save figure" name path err
        end
    end
    return saved_paths
end

function pr_distribution_dataframe(g; bins = -1.5:0.05:1.5)
    P = state(g)
    h = fit(Histogram, vec(P), bins)
    density = h.weights ./ sum(h.weights)
    bin_left = Float64.(h.edges[1][1:end-1])
    return DataFrame(
        bin_left = bin_left,
        bin_center = bin_left .+ step(bins) / 2,
        prob = Float64.(density),
        counts = Float64.(h.weights),
    )
end

function save_pr_distribution_figure(g, outdir, base_name; bins = -1.5:0.05:1.5, enabled = true)
    enabled || return nothing
    df_dist = pr_distribution_dataframe(g; bins)
    fig_dist = Figure()
    ax_dist = Axis(fig_dist[1, 1], xlabel = "P", ylabel = "Probability")
    barplot!(ax_dist, df_dist.bin_left, df_dist.prob; width = step(bins))
    path = joinpath(outdir, base_name * "_Pr_distribution.png")
    try
        save(path, fig_dist)
        return path
    catch err
        @warn "Failed to save Pr distribution figure" path err
        return nothing
    end
end

function excel_cell_value(x)
    ismissing(x) && return missing
    x isa Bool && return x
    x isa Int64 && return x
    x isa Integer && return Int64(x)
    x isa Float64 && return x
    x isa Real && return Float64(x)
    x isa Date && return x
    x isa DateTime && return x
    x isa Time && return x
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

function save_experiment_outputs(; outdir, base_name, g, params, reduced_energy = DataFrame(), landau_coefficients = DataFrame(), save_figures = true, save_xlsx = true)
    mkpath(outdir)
    saved_figures = save_existing_figures(outdir, base_name; enabled = save_figures)
    dist_path = save_pr_distribution_figure(g, outdir, base_name; enabled = save_figures)
    if !isnothing(dist_path)
        push!(saved_figures, dist_path)
    end

    xlsx_path = joinpath(outdir, base_name * ".xlsx")
    if save_xlsx
        df_series = collect_run_series_dataframe()
        df_dist = pr_distribution_dataframe(g)
        if isempty(names(reduced_energy))
            reduced_energy = DataFrame(section = String[], key = String[], value = Any[], note = String[])
        end
        if isempty(names(landau_coefficients))
            landau_coefficients = DataFrame(
                coefficient = String[],
                mean = Float64[],
                std = Float64[],
                minimum = Float64[],
                maximum = Float64[],
                first_value = Float64[],
                n = Int[],
            )
        end
        XLSX.openxlsx(xlsx_path, mode = "w") do xf
            xf[1].name = "series"
            write_dataframe_sheet!(xf["series"], df_series)
            XLSX.addsheet!(xf, "params")
            write_dataframe_sheet!(xf["params"], params)
            XLSX.addsheet!(xf, "reduced_energy")
            write_dataframe_sheet!(xf["reduced_energy"], reduced_energy)
            XLSX.addsheet!(xf, "landau_coefficients")
            write_dataframe_sheet!(xf["landau_coefficients"], landau_coefficients)
            XLSX.addsheet!(xf, "Pr_distribution")
            write_dataframe_sheet!(xf["Pr_distribution"], df_dist)
        end
    end
    return (; xlsx_path, saved_figures)
end

#########################################################################

## ======================== Define simulation ======================== ##
xL = 10  # Length in the x-dimension
yL = 10  # Length in the y-dimension
zL = 10   # Length in the z-dimension

### Preferred naming:
### - `wf_*` = weight-function definition
### - `wg_*` = instantiated generator
### - `dr` already includes `LatticeConstants(...)`
### - `dc` is the integer lattice offset
### wg(; dr = ..., c1 = ..., c2 = ..., dc = ...)， c1 and c2 are coodinates for dipoles
wg_ferro_long = @WG (; dr) -> wf_ferro_inv_r2(; dr) NN = 3
wg_skyrmion = @WG (; dc) -> wf_skyrmion_like(; dc) NN = 3
wg_shell_ferro = @WG (; dc) -> wf_shell_ferro(1, 0.1, 0.1; dc) NN = 3

### Other useful examples:
wg_aniso_nn = @WG (; dc) -> wf_nn_anisotropic(1.0, 0.5; dc) NN = 1
wg_aniso_nn_decay = @WG (; dr, dc) -> wf_nn_anisotropic_inv_r(1.0, 0.5; dr, dc) NN = 1
wg_layered_afe = @WG (; dc) -> wf_layered_afe_nn(1.0, 0.5; dc) NN = 1
wg_shell_competing = @WG (; dc) -> wf_shell_competing(1.0, 0.5, 0.25; dc) NN = 3
# Output directory for the whole sweep
outdir = raw"D:\Code\data\Manuscript\Demo1"
mkpath(outdir)
# ---- parameters to sweep ----
JIsing = 1.0
Scale = 1
Screening = 1
Temp_aneal= 2
Temp = 0.15

a1 = -0.2
b1 = -1.4
c1 = 1
d1 = -1
e1 = 1
linear_field_coeff = 1.0
defect_field_coeff = 0.0

nspins = xL * yL * zL
coeff2 = fill(Float32(a1), nspins)
coeff4 = fill(Float32(b1), nspins)
coeff6 = fill(Float32(c1), nspins)
coeff8 = fill(Float32(d1), nspins)
coeff10 = fill(Float32(e1), nspins)

# Optional spatial disorder examples. Keep them off for the baseline check.
apply_weak_landau_disorder = false
coeff2_disorder_scale = 0.1f0
coeff4_disorder_scale = 0.8f0
coeff6_disorder_scale = 0.5f0
coeff8_disorder_scale = 0.2f0
coeff10_disorder_scale = 0.2f0
if apply_weak_landau_disorder
    coeff2 .+= coeff2_disorder_scale .* randn(Float32, nspins)
    coeff4 .+= coeff4_disorder_scale .* randn(Float32, nspins)
    coeff6 .+= coeff6_disorder_scale .* randn(Float32, nspins)
    coeff8 .+= coeff8_disorder_scale .* randn(Float32, nspins)
    coeff10 .+= coeff10_disorder_scale .* randn(Float32, nspins)
end

proposal_delta = 0.1  # use 0.1, 0.2, 0.5 for LocalProposer(delta)
proposer_args = isnothing(proposal_delta) ? () : (LocalProposer(proposal_delta),)
g = IsingGraph(xL, yL, zL, 
        Continuous(), 
        proposer_args...,
        wg_shell_ferro, 
        LatticeConstants(1.0, 1.0, 1.0),
        # Ising(b = UniformArray(0), localpotential = coeff2) + 
            InteractiveIsing.MagField(b = linear_field_coeff) + InteractiveIsing.Bilinear() + 
            CoulombHamiltonian(scaling = Scale, screening = Screening, recalc = 1000) + 
            Quadratic(localpotential = coeff2) +
            Quartic(localpotential = coeff4) + 
            Sextic(localpotential = coeff6) +
            Octic(localpotential = coeff8) +
            PolynomialHamiltonian(10; localpotential = coeff10), 
        StateSet(-1.5f0, 1.5f0),
        periodic = (:x,:y),
        diag = StateLike(UniformArray)
)
normalize_adj_by_average_col!(g.adj, JIsing)
# Independent Landau: coeff2 carries the quadratic term, so do not set adj(g)[1,1] = a1.
# state(g[1])[1:div(xL,2), :, :] .=  1.0f0
# state(g[1])[div(xL,2)+1:end, :, :] .= -1.0f0
interface(g)

# Temperature init
temp!(g, Temp)

reduced_energy_summary = print_reduced_parameter_summary_reference(;
    a = a1,
    b = b1,
    c = c1,
    d = d1,
    e = e1,
    g,
    JIsing,
    Scale,
    Screening,
    field_typ = linear_field_coeff,
    defect_typ = defect_field_coeff,
    Pmin = -1.5,
    Pmax = 1.5,
)

# ----- Annealing algorithm -----
time_fctr= 1
Steps_1= 4000

Amp1 = 5
nrepeats = 2
pulse1 = TrianglePulseA(Amp1, nrepeats)
pulse2 = SinPulseA(Amp1, nrepeats)
pulse3 = Unique(SinPulseA(Amp1, nrepeats))
AnealingB = LinAnealingB(Temp_aneal, 0f0)
# algorithm_name = :metropolis
# algorithm_kwargs = (;)
algorithm_name = :local_langevin
algorithm_kwargs = (; stepsize = 0.02f0, adjusted = true)
# algorithm_name = :global_langevin
# algorithm_kwargs = (; stepsize = 0.02f0, adjusted = true)
# algorithm_name = :block_langevin
# algorithm_kwargs = (; stepsize = 0.02f0, block_size = 10, adjusted = true)
dynamics = select_dynamics(g, algorithm_name; algorithm_kwargs)


fullsweep = xL*yL*zL
point_repeat = fullsweep*time_fctr
# point_repeat = time_fctr
anneal_time = point_repeat*Steps_1
pulse_time = point_repeat*Steps_1
relax_time = point_repeat*Steps_1/2


capture_interval1 = pulse_time/(nrepeats*4)
capture_interval2 = relax_time/2 
# capture_interval3 = relax_time/2 

M_Integrate_and_Logger = IntegrateAndLog(Float32, point_repeat)
B_Logger = ValueLogger(:b)
T_Logger = ValueLogger(:T)
Depol_Logger = DepolLogger(:depol)
PAFEz_Logger = ValueLogger(:P_AFE_z)
PTop_Logger = ValueLogger(:P_top)
PMid_Logger = ValueLogger(:P_mid)
PBot_Logger = ValueLogger(:P_bot)
Graph_Logger = ImageCapture(:Graph,-1.5,1.5)


# #---- Anealing step ----
# Metro_T = @CompositeAlgorithm begin
#     @alias dynamics = dynamics

#     proposal = @every 1 dynamics()
#     @every 1 M_Integrate_and_Logger(Δvalue = @transform(accepted_proposal_delta_base, proposal))
#     @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
#     @every point_repeat T_Logger(value = @transform(temp, dynamics.model))
#     @every point_repeat Depol_Logger(
#         model = dynamics.model,
#         hamiltonian = dynamics.hamiltonian,
#     )
#     @every point_repeat PAFEz_Logger(value = @transform(staggered_z_polarization, dynamics.model))
#     @every point_repeat PTop_Logger(value = @transform(m -> mean_polarization_zlayer(m, zL), dynamics.model))
#     @every point_repeat PMid_Logger(value = @transform(m -> mean_polarization_zlayer(m, cld(zL, 2)), dynamics.model))
#     @every point_repeat PBot_Logger(value = @transform(m -> mean_polarization_zlayer(m, 1), dynamics.model))
# end
# anneal_partB = @CompositeAlgorithm begin
#     @context metro_t = Metro_T()

#     @every point_repeat AnealingB(model = metro_t.dynamics.model)
# end
# Anealing_step = @Routine begin
#     @repeat anneal_time anneal_partB()
# end

# createProcess(g, Anealing_step, lifetime = 1,
#     Input(M_Integrate_and_Logger, initialvalue = sum(state(g))))
# c = process(g) |> fetch
# voltage1 = c[B_Logger].values
# Pr1 = c[M_Integrate_and_Logger].log
# Temp1 = c[T_Logger].values
# depol_mean1 = c[Depol_Logger].means
# depol_median1 = c[Depol_Logger].medians
# depol_max1 = c[Depol_Logger].maxima
# Htotal1 = c[Depol_Logger].total_energy
# Hdep1 = c[Depol_Logger].depol_energy
# HJ1 = c[Depol_Logger].interaction_energy
# Hfield1 = c[Depol_Logger].field_energy
# Hpoly1 = c[Depol_Logger].poly_energy
# Hrest1 = Hdep1 .+ HJ1 .+ Hpoly1
# PAFEz1 = c[PAFEz_Logger].values
# Ptop1 = c[PTop_Logger].values
# Pmid1 = c[PMid_Logger].values
# Pbot1 = c[PBot_Logger].values

# fTPr = makieaxis(f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Pr"), ax -> lines!(ax, Temp1, Pr1))
# fPr_anneal = makieaxis(f -> Axis(f[1, 1], xlabel = "Step", ylabel = "Pr"), ax -> lines!(ax, Pr1))

# fOrderStep_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Step", ylabel = "Order parameters"),
#     ax -> lines!(ax, PAFEz1, label = "P_AFE_z"),
#     ax -> lines!(ax, Ptop1, label = "P_top"),
#     ax -> lines!(ax, Pmid1, label = "P_mid"),
#     ax -> lines!(ax, Pbot1, label = "P_bot"),
#     ax -> axislegend(ax),
# )

# fOrderT_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Order parameters"),
#     ax -> lines!(ax, Temp1, PAFEz1, label = "P_AFE_z"),
#     ax -> lines!(ax, Temp1, Ptop1, label = "P_top"),
#     ax -> lines!(ax, Temp1, Pmid1, label = "P_mid"),
#     ax -> lines!(ax, Temp1, Pbot1, label = "P_bot"),
#     ax -> axislegend(ax),
# )

# fHrestPr_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Hrest"),
#     ax -> lines!(ax, Pr1, Hrest1),
# )

# fHrestT_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Hrest"),
#     ax -> lines!(ax, Temp1, Hrest1),
# )

# fHtotP_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Total H"),
#     ax -> lines!(ax, Pr1, Htotal1),
# )

# fHtotT_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Total H"),
#     ax -> lines!(ax, Temp1, Htotal1),
# )

# fHtermsPr_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Hamiltonian terms"),
#     ax -> lines!(ax, Pr1, HJ1, label = "H_J"),
#     ax -> lines!(ax, Pr1, Hfield1, label = "H_field"),
#     ax -> lines!(ax, Pr1, Hpoly1, label = "H_poly"),
#     ax -> lines!(ax, Pr1, Hdep1, label = "H_dep"),
#     ax -> axislegend(ax),
# )

# fHtermsT_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Hamiltonian terms"),
#     ax -> lines!(ax, Temp1, HJ1, label = "H_J"),
#     ax -> lines!(ax, Temp1, Hfield1, label = "H_field"),
#     ax -> lines!(ax, Temp1, Hpoly1, label = "H_poly"),
#     ax -> lines!(ax, Temp1, Hdep1, label = "H_dep"),
#     ax -> axislegend(ax),
# )

# fDepP_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Depolarization diagnostic"),
#     ax -> lines!(ax, Pr1, depol_mean1),
#     ax -> lines!(ax, Pr1, depol_median1),
#     ax -> lines!(ax, Pr1, depol_max1),
# )

# fDepT_anneal = makieaxis(
#     f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Depolarization diagnostic"),
#     ax -> lines!(ax, Temp1, depol_mean1),
#     ax -> lines!(ax, Temp1, depol_median1),
#     ax -> lines!(ax, Temp1, depol_max1),
# )



# ----- Pulse Step -----
Metro_Pulse = @CompositeAlgorithm begin
    @alias dynamics = dynamics

    proposal = @every 1 dynamics()
    @every 1 M_Integrate_and_Logger(Δvalue = @transform(accepted_proposal_delta_base, proposal))
    @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
    @every point_repeat Depol_Logger(
        model = dynamics.model,
        hamiltonian = dynamics.hamiltonian,
    )
    @every point_repeat PAFEz_Logger(value = @transform(staggered_z_polarization, dynamics.model))
    @every point_repeat PTop_Logger(value = @transform(m -> mean_polarization_zlayer(m, zL), dynamics.model))
    @every point_repeat PMid_Logger(value = @transform(m -> mean_polarization_zlayer(m, cld(zL, 2)), dynamics.model))
    @every point_repeat PBot_Logger(value = @transform(m -> mean_polarization_zlayer(m, 1), dynamics.model))
end
pulse_part1 = @CompositeAlgorithm begin
    @context metro_pulse = Metro_Pulse()

    @every point_repeat pulse1(
        hamiltonian = metro_pulse.dynamics.hamiltonian,
        M = metro_pulse.dynamics.M,
    )
    # @every capture_interval1 Graph_Logger(array = @transform(model -> state(model), metro_pulse.dynamics.model))
end
relax_part1 = @CompositeAlgorithm begin
    @context metro_pulse = Metro_Pulse()

    # @every capture_interval2 Graph_Logger(array = @transform(model -> state(model), metro_pulse.dynamics.model))
end
Pulse_and_Relax = @Routine begin
    @repeat pulse_time pulse_part1()
    @repeat relax_time relax_part1()
end

# ---- Start simulation2 ----
createProcess(g, Pulse_and_Relax, lifetime = 1, 
    # Input(Graph_Logger, filepath = joinpath(outdir, "capture")),
    Input(M_Integrate_and_Logger, initialvalue = sum(state(g))))
c = process(g) |> fetch

# ---- Collect data2 ----
voltage2 = c[B_Logger].values
Pr2      = c[M_Integrate_and_Logger].log
depol_mean2 = c[Depol_Logger].means
depol_median2 = c[Depol_Logger].medians
depol_max2 = c[Depol_Logger].maxima
Htotal2 = c[Depol_Logger].total_energy
Hdep2 = c[Depol_Logger].depol_energy
HJ2 = c[Depol_Logger].interaction_energy
Hfield2 = c[Depol_Logger].field_energy
Hpoly2 = c[Depol_Logger].poly_energy
Hrest2 = Hdep2 .+ HJ2 .+ Hpoly2
PAFEz2 = c[PAFEz_Logger].values
Ptop2 = c[PTop_Logger].values
Pmid2 = c[PMid_Logger].values
Pbot2 = c[PBot_Logger].values
# Temp1    = c[T_Logger].values

# Set true when you want GLMakie windows to pop up while figures are created.
show_figures = false

fVPr = makieaxis(f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Pr"), ax -> lines!(ax, voltage2, Pr2))
fPr  = makieaxis(f -> Axis(f[1, 1], xlabel = "Step", ylabel = "Pr"), ax -> lines!(ax, Pr2))

fOrderStep = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Step", ylabel = "Order parameters"),
    ax -> lines!(ax, PAFEz2, label = "P_AFE_z"),
    ax -> lines!(ax, Ptop2, label = "P_top"),
    ax -> lines!(ax, Pmid2, label = "P_mid"),
    ax -> lines!(ax, Pbot2, label = "P_bot"),
    ax -> axislegend(ax),
)

fOrderV = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Order parameters"),
    ax -> lines!(ax, voltage2, PAFEz2, label = "P_AFE_z"),
    ax -> lines!(ax, voltage2, Ptop2, label = "P_top"),
    ax -> lines!(ax, voltage2, Pmid2, label = "P_mid"),
    ax -> lines!(ax, voltage2, Pbot2, label = "P_bot"),
    ax -> axislegend(ax),
)


fHrestPr = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Total H"),
    ax -> lines!(ax, Pr2, Hrest2)
)

fHrestV = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Total H"),
    ax -> lines!(ax, voltage2, Hrest2)
)


fHtotP = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Total H"),
    ax -> lines!(ax, Pr2, Htotal2)
)

fHtotV = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Total H"),
    ax -> lines!(ax, voltage2, Htotal2)
)

fHtermsPr = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Hamiltonian terms"),
    ax -> lines!(ax, Pr2, HJ2, label = "H_J"),
    ax -> lines!(ax, Pr2, Hfield2, label = "H_field"),
    ax -> lines!(ax, Pr2, Hpoly2, label = "H_poly"),
    ax -> lines!(ax, Pr2, Hdep2, label = "H_dep"),
    ax -> axislegend(ax),
)

fHtermsV = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Hamiltonian terms"),
    ax -> lines!(ax, voltage2, HJ2, label = "H_J"),
    ax -> lines!(ax, voltage2, Hfield2, label = "H_field"),
    ax -> lines!(ax, voltage2, Hpoly2, label = "H_poly"),
    ax -> lines!(ax, voltage2, Hdep2, label = "H_dep"),
    ax -> axislegend(ax),
)

fDepP = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Pr", ylabel = "Depolarization diagnostic"),
    ax -> lines!(ax, Pr2, depol_mean2),
    ax -> lines!(ax, Pr2, depol_median2),
    ax -> lines!(ax, Pr2, depol_max2),
)

fDepV = makieaxis(
    f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Depolarization diagnostic"),
    ax -> lines!(ax, voltage2, depol_mean2),
    ax -> lines!(ax, voltage2, depol_median2),
    ax -> lines!(ax, voltage2, depol_max2),
)

#######################################################################

# ============ SAVE (PNG + XLSX) ============
save_outputs = true
save_all_figures = true
save_excel = true

route_name = active_route_name()
base_name = make_run_basename(route = route_name)

run_params = params_dataframe(
    "JIsing" => JIsing,
    "a1" => a1,
    "b1" => b1,
    "c1" => c1,
    "d1" => d1,
    "e1" => e1,
    "linear_field_coeff" => linear_field_coeff,
    "defect_field_coeff" => defect_field_coeff,
    "xL" => xL,
    "yL" => yL,
    "zL" => zL,
    "Scale" => Scale,
    "Screening" => Screening,
    "Steps_1" => Steps_1,
    "time_fctr" => time_fctr,
    "anneal_time" => anneal_time,
    "pulse_time" => pulse_time,
    "relax_time" => relax_time,
    "point_repeat" => point_repeat,
    "Temp_aneal" => Temp_aneal,
    "Temp" => Temp,
    "Amp1" => Amp1,
    "nrepeats" => nrepeats,
    "algorithm_name" => string(algorithm_name),
    "algorithm_kwargs" => string(algorithm_kwargs),
    "proposal_delta" => string(proposal_delta),
    "apply_weak_landau_disorder" => apply_weak_landau_disorder,
    "route_name" => route_name,
    "base_name" => base_name,
    "show_figures" => show_figures,
    "save_all_figures" => save_all_figures,
    "save_excel" => save_excel,
    "coeff2_disorder_scale" => coeff2_disorder_scale,
    "coeff4_disorder_scale" => coeff4_disorder_scale,
    "coeff6_disorder_scale" => coeff6_disorder_scale,
    "coeff8_disorder_scale" => coeff8_disorder_scale,
    "coeff10_disorder_scale" => coeff10_disorder_scale,
)

landau_coeff_summary = coefficient_summary_dataframe(
    "P1_MagField" => fill(linear_field_coeff, nspins),
    "P2_Quadratic" => coeff2,
    "P4_Quartic" => coeff4,
    "P6_Sextic" => coeff6,
    "P8_Octic" => coeff8,
    "P10_Polynomial" => coeff10,
)

if save_outputs
    saved = save_experiment_outputs(;
        outdir,
        base_name,
        g,
        params = run_params,
        reduced_energy = reduced_energy_summary,
        landau_coefficients = landau_coeff_summary,
        save_figures = save_all_figures,
        save_xlsx = save_excel,
    )
    println("Saved Excel: ", saved.xlsx_path)
    println("Saved figures:")
    for path in saved.saved_figures
        println("  ", path)
    end
end
# ============ END SAVE (PNG + XLSX) ============



