using Carlo
using Carlo.JobTools
using BenchmarkTools
using Random

mutable struct MC3D <: Carlo.AbstractMC
    T::Float64
    J::Float64
    h::Float64
    spins::Array{Int8,3}
end

function MC3D(params::AbstractDict)
    Lx = Int(params[:Lx])
    Ly = Int(get(params, :Ly, Lx))
    Lz = Int(get(params, :Lz, Lx))

    T = Float64(params[:T])
    J = Float64(get(params, :J, 1.0))
    h = Float64(get(params, :h, 0.0))

    return MC3D(T, J, h, zeros(Int8, Lx, Ly, Lz))
end

@inline _plus1(i::Int, L::Int) = (i == L ? 1 : i + 1)
@inline _minus1(i::Int, L::Int) = (i == 1 ? L : i - 1)

@inline function periodic_elem(spins::Array{Int8,3}, x::Int, y::Int, z::Int)
    Lx, Ly, Lz = size(spins)
    return spins[mod1(x, Lx), mod1(y, Ly), mod1(z, Lz)]
end

@inline function metropolis_step!(mc::MC3D, rng::AbstractRNG)
    Lx, Ly, Lz = size(mc.spins)

    x = rand(rng, 1:Lx)
    y = rand(rng, 1:Ly)
    z = rand(rng, 1:Lz)

    xp = _plus1(x, Lx)
    xm = _minus1(x, Lx)
    yp = _plus1(y, Ly)
    ym = _minus1(y, Ly)
    zp = _plus1(z, Lz)
    zm = _minus1(z, Lz)

    s = mc.spins[x, y, z]
    nn_sum =
        mc.spins[xp, y, z] +
        mc.spins[xm, y, z] +
        mc.spins[x, yp, z] +
        mc.spins[x, ym, z] +
        mc.spins[x, y, zp] +
        mc.spins[x, y, zm]

    dE = 2.0 * s * (mc.J * nn_sum + mc.h)
    if dE <= 0 || rand(rng) < exp(-dE / mc.T)
        mc.spins[x, y, z] = -s
        return 1
    end

    return 0
end

function Carlo.init!(mc::MC3D, ctx::Carlo.MCContext, params::AbstractDict)
    mc.spins .= rand(ctx.rng, Bool, size(mc.spins)) .* 2 .- 1
    return nothing
end

function Carlo.sweep!(mc::MC3D, ctx::Carlo.MCContext)
    for _ = 1:length(mc.spins)
        metropolis_step!(mc, ctx.rng)
    end

    return nothing
end

function run_steps!(mc::MC3D, rng::AbstractRNG, steps::Int)
    accepted = 0
    for i = 1:steps
        accepted += metropolis_step!(mc, rng)
    end
    return accepted
end

function warmup!(mc::MC3D, rng::AbstractRNG, warmup_steps::Int)
    for i = 1:warmup_steps
        metropolis_step!(mc, rng)
    end
    return nothing
end

function _energy(mc::MC3D)
    Lx, Ly, Lz = size(mc.spins)

    e = 0.0
    @inbounds for z = 1:Lz, y = 1:Ly, x = 1:Lx
        s = mc.spins[x, y, z]
        e -= mc.J * s * (
            mc.spins[_plus1(x, Lx), y, z] +
            mc.spins[x, _plus1(y, Ly), z] +
            mc.spins[x, y, _plus1(z, Lz)]
        )
        e -= mc.h * s
    end

    return e
end

function Carlo.measure!(mc::MC3D, ctx::Carlo.MCContext)
    n = length(mc.spins)

    e = _energy(mc)
    m = sum(mc.spins)

    e_per_spin = e / n
    m_per_spin = m / n

    Carlo.measure!(ctx, :Energy, e_per_spin)
    Carlo.measure!(ctx, :Energy2, e_per_spin^2)
    Carlo.measure!(ctx, :Magnetization, m_per_spin)
    Carlo.measure!(ctx, :AbsMagnetization, abs(m_per_spin))
    Carlo.measure!(ctx, :Magnetization2, m_per_spin^2)
    Carlo.measure!(ctx, :Magnetization4, m_per_spin^4)

    return nothing
end

function Carlo.register_evaluables(
    ::Type{MC3D},
    eval::Carlo.AbstractEvaluator,
    params::AbstractDict,
)
    T = Float64(params[:T])
    Lx = Int(params[:Lx])
    Ly = Int(get(params, :Ly, Lx))
    Lz = Int(get(params, :Lz, Lx))
    n = Lx * Ly * Lz

    Carlo.evaluate!(eval, :BinderRatio, (:Magnetization2, :Magnetization4)) do m2, m4
        return (m2 * m2) / m4
    end

    Carlo.evaluate!(eval, :SpecificHeat, (:Energy2, :Energy)) do e2, e
        return n * (e2 - e^2) / (T^2)
    end

    Carlo.evaluate!(eval, :Susceptibility, (:Magnetization2, :AbsMagnetization)) do m2, abs_m
        return n * (m2 - abs_m^2) / T
    end

    return nothing
end

# Keep checkpointing compatible without introducing direct HDF5 dependency here.
Carlo.write_checkpoint(::MC3D, out) = nothing
Carlo.read_checkpoint!(::MC3D, in) = nothing

function benchmark_steps(;
    L::Int = 20,
    T::Float64 = 4.5,
    J::Float64 = 1.0,
    h::Float64 = 0.0,
    steps::Int = 1_000_000,
    seed::Int = 42,
    warmup_steps::Int = 20_000,
    samples::Int = 10,
    evals::Int = 1,
)
    benchmark = @benchmarkable run_steps!(mc, rng, $steps) setup = begin
        params = Dict{Symbol,Any}(
            :Lx => $L,
            :Ly => $L,
            :Lz => $L,
            :T => $T,
            :J => $J,
            :h => $h,
            :binsize => 1,
            :thermalization => 0,
            :seed => $seed,
        )
        mc = MC3D(params)
        ctx = Carlo.MCContext{Random.Xoshiro}(params)
        Carlo.init!(mc, ctx, params)
        rng = ctx.rng
        warmup!(mc, rng, $warmup_steps)
    end evals = evals samples = samples

    return run(benchmark)
end

function benchmark_summary(
    trial::BenchmarkTools.Trial;
    steps::Int = 1_000_000,
)
    min_ns = minimum(trial).time
    median_ns = median(trial).time
    mean_ns = mean(trial).time

    return (
        steps = steps,
        min_ns_per_step = min_ns / steps,
        median_ns_per_step = median_ns / steps,
        mean_ns_per_step = mean_ns / steps,
        min_steps_per_second = 1e9 * steps / min_ns,
        median_steps_per_second = 1e9 * steps / median_ns,
        mean_steps_per_second = 1e9 * steps / mean_ns,
    )
end

function benchmark_1m(; kwargs...)
    trial = benchmark_steps(steps = 1_000_000; kwargs...)
    return trial, benchmark_summary(trial; steps = 1_000_000)
end

function default_job(; L::Int = 10, T::Float64 = 4.5)
    tm = TaskMaker()
    tm.sweeps = 10_000
    tm.thermalization = 2_000
    tm.binsize = 20

    tm.Lx = L
    tm.Ly = L
    tm.Lz = L
    tm.J = 1.0
    tm.h = 0.0

    task(tm; T = T)

    return JobInfo(
        splitext(@__FILE__)[1],
        MC3D;
        checkpoint_time = "20:00",
        run_time = "00:10:00",
        tasks = make_tasks(tm),
    )
end
