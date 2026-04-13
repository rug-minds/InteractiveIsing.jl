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

function newmakie(makietype, args...; kwargs...)
    f = makietype(args...; kwargs...)
    scr = GLMakie.Screen()
    display(scr, f)
    f
end

function makieaxis(axisfunc, modifiers...)
    f = Figure()
    ax = axisfunc(f[1, 1])
    for mod in modifiers
        mod(ax)
    end
    scr = GLMakie.Screen()
    display(scr, f)
    f
end

# Weight function variant 1
function weightfunc1(; dc::T) where {T}
    prefac = 1
    d = dc
    # Always positive coupling (ferromagnetic)
    return prefac / norm2(d)
end
function weightfunc2(; dc)
    d = dc
    dx, dy, dz = d  # 先解包
    physical_dr2 = sqrt((0.05*dx)^2 + (0.05*dy)^2 + (0.2*dz)^2) 
    # z 方向保持铁磁 (正耦合)
    # if dx == 0 && dy == 0
    #     prefac = 1
    # elseif dx == 0
    #     prefac = 1
    # else
    #     # xy 平面反铁磁 (负耦合)
    #     prefac = -1
    # end
    prefac = 1
    return prefac / physical_dr2
end
function weightfunc3(; dc)
    d = dc
    dx, dy, dz = d  # 先解包
    physical_dr2 = sqrt((0.3*dx)^2 + (0.3*dy)^2 + (0.3*dz)^2) 
    # z 方向保持铁磁 (正耦合)
    if dx == 0 && dy == 0
        prefac = 1
    elseif dx == 0
        prefac = 1
    else
        # xy 平面反铁磁 (负耦合)
        prefac = -1
    end
    # prefac = 1
    return prefac / physical_dr2
end
function weightfunc_angle_anti(; dc::DC) where DC
    d = dc
    dx, dy, dz = d  # 先解包
    ax=0.2
    ay=0.2
    az=0.1
    rx = ax*dx
    ry = ay*dy
    rz = az*dz

    r2 = rx^2 + ry^2 + rz^2
    r  = sqrt(r2)

    cosθ = rz / r              # 与 z 轴夹角的 cos
    prefac  = -1 + 3*cosθ^2        # Ising 沿 z 的角度因子

    return prefac / r^3
end
function weightfunc_angle_ferro(; dc)
    d = dc
    dx, dy, dz = d  # 先解包
    ax=0.2
    ay=0.2
    az=0.1
    rx = ax*dx
    ry = ay*dy
    rz = az*dz

    r2 = rx^2 + ry^2 + rz^2
    r  = sqrt(r2)

    cosθ = rz / r              # 与 z 轴夹角的 cos
    prefac  = -1 + 3*cosθ^2        # Ising 沿 z 的角度因子

    return abs(prefac) / r^3
end
# Shell-based coupling + dipolar coupling
function weightfunc_shell(ax, ay, az, csr, lambda1, lambda2; dc)
    dx, dy, dz = dc
    k1  = 1.0
    k2  = lambda1 * k1
    k3  = lambda2 * k2

    # --- physical distance for dipolar term ---
    rx = ax * dx
    ry = ay * dy
    rz = az * dz
    r2 = rx^2 + ry^2 + rz^2

    if r2 == 0
        return 0.0
    end
    r  = sqrt(r2)

    # --- dipolar angular factor (Ising along z) ---
    cosθ  = rz / r
    prefac_dip = -1 + 3 * cosθ^2
    Jdip = prefac_dip / r^3

    # --- shell-based short-range term ---
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

    Jsr = csr * prefac_sr
    # return Jdip + Jsr
    return Jsr
end
# Skymion-like coupling
function weightfunc_skymion(; dc)
    d = dc
    dx, dy, dz = d  # 先解包
    # z 方向保持铁磁 (正耦合)
    prefac = 2
    if abs(dy) > 0 || abs(dx) > 0
        prefac = -2
    end
    
    return prefac / norm2(d)
end

function weightfunc_xy_antiferro(ax, ay, az; dc)
    d = dc
    dx, dy, dz = d  # 先解包
    physical_dr2 = sqrt((ax*dx)^2 + (ay*dy)^2 + (az*dz)^2) 
    # z 方向保持铁磁 (正耦合)
    if dx == 0 && dy == 0
        prefac = 1
    elseif dx == 0
        prefac = 1
    else
        # xy 平面反铁磁 (负耦合)
        prefac = -1
    end
    
    return prefac / physical_dr2
end

function weightfunc_xy_dilog_antiferro(; dc)
    d = dc
    dx, dy, dz = d
    
    if (abs(dx) + abs(dy)) % 2 == 0
        return 1.0 / norm2(d)    # 铁磁
    else
        return -1.0 / norm2(d)   # 反铁磁
    end
    
    return prefac / norm2(d)
end

function weightfunc4(; dc)
    prefac = -1
    d = dc
    # Always positive coupling (ferromagnetic)
    return prefac / norm2(d)
end

##################################################################################
### struct start: TrianglePulseA (simple four-segment triangular waveform)
### Run with TrianlePulseA
###  /\
### /  \    _____
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
    (;current_T, dT, state) = context
    temp!(state, max(current_T, 0))
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
    (;tem_pulse, step, state) = context

    temval = tem_pulse[step]

    temp!(state, max(temval, 0))

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
    c = CompositeAlgorithm(integrator, logger, (1, loginterval), Route(integrator => logger, :total => :value, transform = x -> x[]))
    pack = package(c)
end

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

xL = 40  # Length in the x-dimension
yL = 40  # Length in the y-dimension
zL = 10   # Length in the z-dimension


### weightfunc_shell(dr,c1,c2, ax, ay, az, csr, lambda1, lambda2), Lambda is the ratio between different shells
wg1 = @WG (; dc) -> weightfunc1(; dc) NN = 3
wg2 = @WG (; dc) -> weightfunc_skymion(; dc) NN = 3
wg5 = @WG (; dc) -> weightfunc_shell(1, 1, 1, 1, 0.1, 0.1; dc) NN = 3

a1, c1 = -2, 10
b1 =-(a1+3*c1)/2
Ex = range(-1.5, 1.5, length=1000)
Ey = a1 .* Ex.^2 .+ b1 .* Ex.^4 .+ c1 .* Ex.^6
f1=newmakie(lines, Ex, Ey);

E_barrier= abs(a1 * 1^2 + b1 * 1^4 .+ c1 * 1^6)
println("E_barrier = ", E_barrier)
Epp_1 = 2a1 + 12b1 + 30c1   # Ey''(1)
println("Ey''(1) = ", Epp_1)

# Output directory for the whole sweep
outdir = raw"D:\Code\data\Manuscript\Demo1"
mkpath(outdir)

# Run a sweep without re-running the first two setup cells.
# This cell will reconfigure the Hamiltonian, run the process, and SAVE (PNG + XLSX) for each Screening.

# function run_one_screening!(g; Scale, Screening, Temp_aneal=6f0, time_fctr=30, Steps_1=6000)
# ---- parameters to sweep ----
Scale = 1
Screening = 0.01   # <-- change range here
Temp_aneal=5f0
time_fctr=1
Steps_1=6000
Temp = 0.5

#### 可以给Quartic写个vector，然后后面就不哦那个给
#### b=:homogeneous会被移除，换成b=:OffsetArray, UniformArray, ConstFill，（ConstValue = ConstFill with dimension 0）

#=
现在我们可以分别使用c,localpotential 在ising，quartic，sextic里设置不同参数。
    如果使用默认数值，那多次项会和Jii耦合在一起。如果Jii是2， 后面的多次项相当于都含有一个2.
    调整参数的时候要注意
=#
#=
如果使用这个方式，Ising项的c参数和localpotential项的参数就不会耦合在一起了，可以独立调整。
    g = IsingGraph(xL, yL, zL, 
        Continuous(), 
        wg5, 
        LatticeConstants(1.0, 1.0, 1.0),
        Ising(b = StateLike(UniformArray,0), localpotential = StateLike(UniformArray,0)) + 
            CoulombHamiltonian(scaling = Scale, screening = Screening, recalc = 1000) + 
            Quartic(localpotential = StateLike(UniformArray,0)) + 
            Sextic(localpotential = StateLike(UniformArray,0)), 
        StateSet(-1.5f0, 1.5f0),
        periodic = (:x,:y),
        diag = StateLike(UniformArray)
    )
    ###这样的话，可以在后续直接调整每一个参数。
        g.hamiltonian[1].lp[] = a1
        g.hamiltonian[1].c[] = 1
        g.hamiltonian[5].lp[] = b1
        g.hamiltonian[5].c[] = 1
        g.hamiltonian[6].lp[] = c1
        g.hamiltonian[6].c[] = 1
=#


g = IsingGraph(xL, yL, zL, 
        Continuous(), 
        wg5, 
        LatticeConstants(1.0, 1.0, 1.0),
        Ising(b = StateLike(UniformArray,0)) + 
            CoulombHamiltonian(scaling = Scale, screening = Screening, recalc = 1000) + 
            Quartic(c=b1/a1) + 
            Sextic(c=c1/a1), 
        StateSet(-1.5f0, 1.5f0),
        periodic = (:x,:y),
        diag = StateLike(UniformArray)
)
normalize_adj_by_average_col!(g.adj, 1f0)
adj(g)[1,1] = a1

interface(g)

# reinit(g)

# Temperature init
temp!(g, Temp)

# ----- Annealing algorithm -----
Amp1 =10
nrepeats = 2
pulse1 = TrianglePulseA(Amp1, nrepeats)
pulse2 = SinPulseA(Amp1, nrepeats)
pulse3 = Unique(SinPulseA(Amp1, nrepeats))
AnealingB = LinAnealingB(Temp_aneal, 0f0)
metropolis = g.default_algorithm

fullsweep = xL*yL*zL
anneal_time = time_fctr*fullsweep*Steps_1
pulse_time = time_fctr*fullsweep*Steps_1
relax_time = time_fctr/2*fullsweep*Steps_1
point_repeat = fullsweep*time_fctr

capture_interval1 = pulse_time/(nrepeats*4)
capture_interval2 = relax_time/2 

M_Integrate_and_Logger = IntegrateAndLog(Float32, point_repeat)
B_Logger = ValueLogger(:b)
T_Logger = ValueLogger(:T)
Graph_Logger = ImageCapture(:Graph,-1.5,1.5)

# Metro_T = CompositeAlgorithm(metropolis, M_Integrate_and_Logger, B_Logger, T_Logger,
#     (1, 1, point_repeat, point_repeat),
#     Route(metropolis => M_Integrate_and_Logger, :proposal => :Δvalue, transform = proposal -> accepteddelta(proposal)),
#     Route(metropolis => B_Logger, :hamiltonian => :value, transform = x -> x.b[]),
#     Route(metropolis => T_Logger, :state => :value, transform = temp)
# )
# anneal_partB = CompositeAlgorithm(Metro_T, AnealingB,
#     (1, point_repeat),
#     Route(metropolis => AnealingB, :state),
# )
# Anealing_step = Routine(anneal_partB, (anneal_time,))

# createProcess(g, Anealing_step, lifetime = 1)
# c = process(g) |> fetch
# voltage1 = c[B_Logger].values
# Pr1      = c[M_Integrate_and_Logger].log
# Temp1    = c[T_Logger].values


Metro_Pulse = CompositeAlgorithm(metropolis, M_Integrate_and_Logger, B_Logger,
    (1, 1, point_repeat),
    Route(metropolis => M_Integrate_and_Logger, :proposal => :Δvalue, transform = proposal -> accepteddelta(proposal)),
    Route(metropolis => B_Logger, :hamiltonian => :value, transform = x -> x.b[]),
)

Metro_Pulse = @CompositeAlgorithm begin
    @alias metropolis = metropolis

    proposal = metropolis()
    M_Integrate_and_Logger(Δvalue = accepteddelta(proposal))
    @every point_repeat B_Logger(hamiltonian = metropolis.hamiltonian)
end

pulse_part1 = CompositeAlgorithm(Metro_Pulse, pulse1, Graph_Logger, (1, point_repeat, capture_interval1), 
    Route(metropolis => Graph_Logger, :state => :array, transform = x -> state(x))
)
relax_part1 = CompositeAlgorithm(Metro_Pulse, Graph_Logger, (1, capture_interval2), 
    Route(metropolis => Graph_Logger, :state => :array, transform = x -> state(x))
)
Pulse_and_Relax = Routine(pulse_part1, relax_part1,
    (pulse_time, relax_time),
    Route(metropolis => pulse1, :hamiltonian, :M),
)
createProcess(g, Pulse_and_Relax, lifetime = 1, 
    Input(Graph_Logger, filepath = joinpath(outdir, "capture")),
    Input(M_Integrate_and_Logger, initialvalue = sum(state(g))))
c = process(g) |> fetch
voltage2 = c[B_Logger].values
Pr2      = c[M_Integrate_and_Logger].log
# Temp1    = c[T_Logger].values


fVPr = makieaxis(f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Pr"), ax -> lines!(ax, voltage2, Pr2))
fPr  = makieaxis(f -> Axis(f[1, 1], xlabel = "Step", ylabel = "Pr"), ax -> lines!(ax, Pr2))





# ============ SAVE (PNG + XLSX) ============
date_str = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
base_name = string(
    "Scale=", round(Scale, digits=4),
    "_Screening=", round(Screening, digits=4),
    "_timefctr=", round(time_fctr, digits=4),
    "_Steps_1=", round(Steps_1, digits=4),
    "_Eb=", round(E_barrier, digits=4),
    "_Epp=", round(Epp_1, digits=4),
    "_Temp_aneal=", round(Temp_aneal, digits=4),
    "_", date_str
)

png_path  = joinpath(outdir, base_name * ".png")
xlsx_path = joinpath(outdir, base_name * ".xlsx")

# # Figure: Temperature vs Pr
# fTPr = makieaxis(f -> Axis(f[1, 1], xlabel = "Temperature", ylabel = "Pr"), ax -> lines!(ax, Temp1, Pr1))
# try
#     save(png_path, fTPr)
#     println("Saved figure: ", png_path)
# catch err
#     @warn "Failed to save figure" err
# end

# Pr distribution histogram
P = state(g[])
v = vec(P)
bins = -1.5:0.05:1.5
h = fit(Histogram, v, bins)
density = h.weights ./ sum(h.weights)

fig_dist = Figure()
ax_dist = Axis(fig_dist[1, 1], xlabel="P", ylabel="Probability")
barplot!(ax_dist, h.edges[1][1:end-1], density; width = step(bins))

png_path_dist = joinpath(outdir, base_name * "_Pr_distribution.png")
try
    save(png_path_dist, fig_dist)
    println("Saved Pr distribution figure: ", png_path_dist)
catch err
    @warn "Failed to save Pr distribution figure" err
end

# # Excel: series + distribution + params
# n = min(length(Temp1), length(Pr1))
# df_series = DataFrame(Temp1 = Float64.(Temp1[1:n]), Pr = Float64.(Pr1[1:n]))

# bin_left = Float64.(h.edges[1][1:end-1])
# bin_center = bin_left .+ step(bins)/2
# df_dist = DataFrame(
#     bin_left   = bin_left,
#     bin_center = bin_center,
#     prob       = Float64.(density),
#     counts     = Float64.(h.weights)
# )

# params = DataFrame(
#     key = String[
#         "JIsing","a1","b1","c1","E_barrier","Eypp_1","xL","yL","zL","Scale","Screening","Steps_1","time_fctr",
#         "anneal_time","point_repeat","Temp_aneal"
#     ],
#     value = Any[
#         JIsing, a1, b1, c1, E_barrier, Epp_1, xL, yL, zL, Scale, Screening, Steps_1, time_fctr,
#         anneal_time, point_repeat, Temp_aneal
#     ]
# )

# XLSX.openxlsx(xlsx_path, mode="w") do xf
#     xf[1].name = "series"
#     XLSX.writetable!(xf["series"], collect(eachcol(df_series)), names(df_series))

#     XLSX.addsheet!(xf, "Pr_distribution")
#     XLSX.writetable!(xf["Pr_distribution"], collect(eachcol(df_dist)), names(df_dist))

#     XLSX.addsheet!(xf, "params")
#     XLSX.writetable!(xf["params"], collect(eachcol(params)), names(params))
# end
# println("Saved Excel: ", xlsx_path)

    # # Return only small metadata to avoid memory blow-up
    # return (; Screening = Float64(Screening), png_path, png_path_dist, xlsx_path)
# end

# # ---- parameters to sweep ----
# Scale = 2
# Screening_values = 15   # <-- change range here

# # ---- loop over Screening ----
# results = Dict{Float64, Any}()
# for Screening in Screening_values
#     @info "Running sweep" Screening
#     res = run_one_screening!(g; Scale=Scale, Screening=Float64(Screening))
#     results[Float64(Screening)] = res
# end

# println("Done. Available keys(Screening) = ", collect(keys(results)))
