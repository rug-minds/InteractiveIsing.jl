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
function weightfunc1(dr,c1,c2)
    prefac = 1
    d = delta(c1,c2)
    dx, dy, dz = d
    # Always positive coupling (ferromagnetic)
    return prefac / norm2(d)
end
function weightfunc2(dr, c1, c2)
    d = delta(c1, c2)
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
function weightfunc3(dr, c1, c2)
    d = delta(c1, c2)
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
function weightfunc_angle_anti(dr, c1, c2)
    d = delta(c1, c2)
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
function weightfunc_angle_ferro(dr, c1, c2)
    d = delta(c1, c2)
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
function weightfunc_shell(dr, c1, c2, ax, ay, az, csr, lambda1, lambda2)
    dx, dy, dz = delta(c1, c2)
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
function weightfunc_skymion(dr,c1,c2)
    d = delta(c1, c2)
    dx, dy, dz = d  # 先解包
    # z 方向保持铁磁 (正耦合)
    prefac = 2
    if abs(dy) > 0 || abs(dx) > 0
        prefac = -2
    end
    
    return prefac / norm2(d)
end

function weightfunc_xy_antiferro(dr, c1, c2, ax, ay, az)
    d = delta(c1, c2)
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

function weightfunc_xy_dilog_antiferro(dr, c1, c2)
    d = delta(c1, c2)
    dx, dy, dz = d
    
    if (abs(dx) + abs(dy)) % 2 == 0
        return 1.0 / norm2(d)    # 铁磁
    else
        return -1.0 / norm2(d)   # 反铁磁
    end
    
    return prefac / norm2(d)
end

function weightfunc4(dr,c1,c2)
    prefac = -1
    d = delta(c1,c2)
    dx, dy, _ = d
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
    (;current_T, dT, isinggraph) = context
    temp(isinggraph, max(current_T, 0))
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
    (;tem_pulse, step, isinggraph) = context

    temval = tem_pulse[step]

    temp(isinggraph, max(temval, 0))

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


xL = 30  # Length in the x-dimension
yL = 30  # Length in the y-dimension
zL = 10   # Length in the z-dimension
g = IsingGraph(xL, yL, zL, stype = Continuous(),periodic = (:x,:y), set = (-1.5,1.5))
# Visual marker size (tune for clarity vs performance)
II.makie_markersize[] = 0.3
# Launch interactive visualization (idle until createProcess(...) later)
interface(g)
g.hamiltonian = sethomogeneousparam(g.hamiltonian, :b)

JIsing = 1.0

#### Weight function setup (Connection setup)

### weightfunc_shell(dr,c1,c2, ax, ay, az, csr, lambda1, lambda2), Lambda is the ratio between different shells
wg5 = @WG (dr,c1,c2) -> weightfunc_shell(dr, c1, c2, 1, 1, 1, JIsing, 0.1, 0.1) NN = 3
genAdj!(g, wg5)

# a1, b1, c1 = -20, 16, 0 
a1, c1 = -2, 10
b1 =-(a1+3*c1)/2
Ex = range(-1.5, 1.5, length=1000)
Ey = a1 .* Ex.^2 .+ b1 .* Ex.^4 .+ c1 .* Ex.^6
f1 = newmakie(lines, Ex, Ey);
E_barrier = abs(a1 * 1^2 + b1 * 1^4 .+ c1 * 1^6)
println("E_barrier = ", E_barrier)
Epp_1 = 2a1 + 12b1 + 30c1   # Ey''(1)
println("Ey''(1) = ", Epp_1)



# Output directory for the whole sweep
outdir = raw"C:\Users\P317151\Documents\data\Model_V1.0\20260217\Pulse sweep\Diff_distance"
mkpath(outdir)

# Run a sweep without re-running the first two setup cells.
# This cell will reconfigure the Hamiltonian, run the process, and SAVE (PNG + XLSX) for each Screening.

function run_simu!(g; X_dis, Y_dis, Z_dis, Scale, Screening, Amp1, Temp=0.3, time_fctr=1, Steps_1=4000)
    # ----- Hamiltonian -----
    setdist!(g, (X_dis,Y_dis,Z_dis))
    g.hamiltonian = Ising(g) + CoulombHamiltonian(g, Scale, screening = Screening) + Quartic(g) + Sextic(g)
    g.hamiltonian = sethomogeneousparam(g.hamiltonian, :b)

    # Landau/self terms
    homogeneousself!(g, a1)
    g.hamiltonian[4].qc[] = b1/a1
    g.hamiltonian[5].sc[] = c1/a1
    g.hamiltonian = sethomogeneousparam(g.hamiltonian, :b)

    # Temperature init
    temp(g, Temp)

    # ----- Pulse algorithm -----
    pulse1 = TrianglePulseA(Amp1, 2)
    metropolis = g.default_algorithm

    M_Logger = ValueLogger(:M)
    B_Logger = ValueLogger(:b)

    fullsweep = xL*yL*zL
    pulse_time = time_fctr*fullsweep*Steps_1
    relax_time = 1/2*time_fctr*fullsweep*Steps_1
    point_repeat = fullsweep*time_fctr

    Metro_and_recal = CompositeAlgorithm(metropolis, Recalc(3), M_Logger, B_Logger,
        (1, 1000, point_repeat, point_repeat),
        Route(metropolis => M_Logger, :M => :value),
        Route(metropolis => B_Logger, :hamiltonian => :value, transform = x -> x.b[]),
        Route(metropolis => Recalc(3), :hamiltonian),
    )

    pulse_part1 = CompositeAlgorithm(Metro_and_recal, pulse1, (1, point_repeat))

    Pulse_and_Relax = Routine(pulse_part1, Metro_and_recal, 
    (pulse_time, relax_time), 
    Route(metropolis => pulse1, :hamiltonian, :M),     
    )

    createProcess(g, Pulse_and_Relax, lifetime = 1)
    c = process(g) |> fetch

    Voltage1 = c[B_Logger].values
    Pr1      = c[M_Logger].values

    # ============ SAVE (PNG + XLSX) ============
    date_str = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    base_name = string(
        "X", round(X_dis, digits=4),
        "_Y=", round(Y_dis, digits=4),
        "_Z=", round(Z_dis, digits=4),
        "Scale=", round(Scale, digits=4),
        "_Screening=", round(Screening, digits=4),
        "_Temp", round(Temp, digits=4),
        "_timefctr=", round(time_fctr, digits=4),
        "_Steps_1=", round(Steps_1, digits=4),
        "_", date_str
    )

    png_path  = joinpath(outdir, base_name * ".png")
    xlsx_path = joinpath(outdir, base_name * ".xlsx")

    # Figure: Voltage vs Pr
    fTPr = makieaxis(f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Pr"), ax -> lines!(ax, Voltage1, Pr1))
    try
        save(png_path, fTPr)
        println("Saved figure: ", png_path)
    catch err
        @warn "Failed to save figure" err
    end

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

    # Excel: series + distribution + params
    df_series = DataFrame(Voltage = Float64.(Voltage1), Pr = Float64.(Pr1))

    bin_left = Float64.(h.edges[1][1:end-1])
    bin_center = bin_left .+ step(bins)/2
    df_dist = DataFrame(
        bin_left   = bin_left,
        bin_center = bin_center,
        prob       = Float64.(density),
        counts     = Float64.(h.weights)
    )

    Temp = Temp

    params = DataFrame(
        key = String[
            "JIsing","a1","b1","c1","E_barrier","Eypp_1","xL","yL","zL","Scale","Screening","Steps_1","time_fctr",
            "pulse_time","relax_time","point_repeat","Temp_init"
        ],
        value = Any[
            JIsing, a1, b1, c1, E_barrier, Epp_1, xL, yL, zL, Scale, Screening, Steps_1, time_fctr,
            pulse_time, relax_time, point_repeat, Temp
        ]
    )

    XLSX.openxlsx(xlsx_path, mode="w") do xf
        xf[1].name = "series"
        XLSX.writetable!(xf["series"], collect(eachcol(df_series)), names(df_series))

        XLSX.addsheet!(xf, "Pr_distribution")
        XLSX.writetable!(xf["Pr_distribution"], collect(eachcol(df_dist)), names(df_dist))

        XLSX.addsheet!(xf, "params")
        XLSX.writetable!(xf["params"], collect(eachcol(params)), names(params))
    end
    println("Saved Excel: ", xlsx_path)

    # Return only small metadata to avoid memory blow-up
    return (; Screening = Float64(Screening), png_path, png_path_dist, xlsx_path)
end

# ---- parameters to sweep ----
Scale = 1
X_dis, Y_dis = 1.0, 1.0
Amp1 = 20
T = 1
Screening = 10   # <-- change range here
Z_dis_values = 0.1:0.2:0.5

# ---- loop over Screening ----
results = Dict{Float64, Any}()
for Z_dis in Z_dis_values
    res = run_simu!(g; X_dis = X_dis, Y_dis = Y_dis, Z_dis=Float64(Z_dis), Amp1=Amp1, Scale=Scale, Temp=T, Screening=Screening)
    results[Float64(Z_dis)] = res
end
println("Done. Available keys(Z_dis) = ", collect(keys(results)))