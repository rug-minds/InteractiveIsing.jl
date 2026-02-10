using InteractiveIsing, GLMakie, FileIO, CairoMakie
using InteractiveIsing.Processes
import InteractiveIsing as II

## Utility functions for experiments
### Use ii. to check if the terms are correct
### Now the H is written like H_self + H_quartic
### Which is Jii*Si^2 + Qc*Jii*Si^4 wichi means Jii=a, Qc*Jii=b in a*Si^2 + b*Si^4

function newmakie(makietype, args...)
    f = makietype(args...)
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
function Processes.prepare(tp::TrianglePulseA, args)
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
### struct start: SinPulseA (simple sine waveform)
### Run with SinPulseA

struct SinPulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end    
function Processes.prepare(tp::SinPulseA, args)
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
function Processes.prepare(tp::LinAnealingA, args)
    n_calls = num_calls(args)
    dT = (tp.stop_T - tp.start_T) / n_calls
    (;current_T = tp.start_T, dT)
end
function Processes.step!(::LinAnealingA, context::C) where C
    (;current_T, dT, isinggraph) = context
    temp(isinggraph, current_T)
    return (;current_T = current_T + dT)
end
##################################################################################

##################################################################################
struct ValueLogger{Name} <: ProcessAlgorithm end
ValueLogger(name) = ValueLogger{Symbol(name)}()
function Processes.prepare(::ValueLogger, args)
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
g = IsingGraph(xL, yL, zL, stype = Continuous(),periodic = (:x,:y))
# Visual marker size (tune for clarity vs performance)
II.makie_markersize[] = 0.3
# Launch interactive visualization (idle until createProcess(...) later)
interface(g)
g.hamiltonian = sethomogeneousparam(g.hamiltonian, :b)

# a1, b1, c1 = -20, 16, 0 
a1, b1, c1 = 0, 0, 0 
Ex = range(-1.0, 1.0, length=1000)
Ey = a1 .* Ex.^2 .+ b1 .* Ex.^4 .+ c1 .* Ex.^6

#### Weight function setup (Connection setup)
#### Set the distance scaling
setdist!(g, (1.0,1.0,1.0))

### weightfunc_shell(dr,c1,c2, ax, ay, az, csr, lambda1, lambda2), Lambda is the ratio between different shells
wg5 = @WG (dr,c1,c2) -> weightfunc_shell(dr, c1, c2, 1, 1, 1, 1, 0.1, 0.1) NN = 3
# wg1 = @WG weightfunc1 NN = (2,2,2)
# wg1 = @WG weightfunc1 NN = (2,2,2)
# wg1 = @WG (dr,c1,c2) -> weightfunc_xy_antiferro(dr, c1, c2, 2, 2, 2) NN = (2,2,2)

genAdj!(g, wg5)



### Set hamiltonian with selfenergy and depolarization field
# CoulombHamiltonian2(g::AbstractIsingGraph, eps::Real = 1.f0; screening = 0.0)
g.hamiltonian = Ising(g) + CoulombHamiltonian2(g, 4, screening = 0.1)

# Only necessary if the Hamiltonian has non-local terms that need to be recalculated after each spin flip.
# reprepare(g)

### Use ii. to check if the terms are correct
### Now the H is written like H_self + H_quartic
### Which is Jii*Si^2 + Qc*Jii*Si^4 wichi means Jii=a, Qc*Jii=b in a*Si^2 + b*Si^4

### Set Jii
g.hamiltonian = sethomogeneousparam(g.hamiltonian, :b)
homogeneousself!(g,a1)

Temperature=1
temp(g,Temperature)

### Run simulation process
fullsweep = xL*yL*zL
time_fctr = 1
anneal_time = fullsweep*5000
pulsetime = fullsweep*5000
relaxtime = fullsweep*5000
point_repeat = time_fctr*fullsweep
pulse1 = TrianglePulseA(20, 2)
pulse2 = SinPulseA(20, 1)
pulse3 = Unique(SinPulseA(5, 1))

Anealing1 = LinAnealingA(2f0, 1f0)
metropolis = g.default_algorithm

#
M_Logger = ValueLogger(:M)
# Pulse_logger = ValueLogger(:pulse)
B_Logger = ValueLogger(:b)




Metro_and_recal = CompositeAlgorithm((metropolis, Recalc(3), M_Logger, B_Logger), (1,200, fullsweep, fullsweep))

pulse_part1 = CompositeAlgorithm((Metro_and_recal, pulse1), (1, point_repeat))
pulse_part2 = CompositeAlgorithm((Metro_and_recal, pulse2, ), (1, point_repeat))
anneal_part1 = CompositeAlgorithm((Metro_and_recal, Anealing1), (1, point_repeat))

Pulse_and_Relax = Routine((pulse_part1, Metro_and_recal), 
    (pulsetime, relaxtime), 
    Route(metropolis, pulse1, :hamiltonian, :M),     
    Route(metropolis, M_Logger, :M => :value), 
    Route(metropolis, B_Logger, :hamiltonian => :value, transform = x -> x.b[]), 
    Route(metropolis, Recalc(3), :hamiltonian),
    )
createProcess(g, Pulse_and_Relax, lifetime = 1)

# getcontext(g)
# getcontext(g)[pulse1]

### estimate time
# est_remaining(process(g))
# Wait until it is done
c = process(g) |> fetch # If you want to close ctr+c
voltage1= c[B_Logger].values
Pr1= c[M_Logger].values

# Voltage2 = c[pulse2].x
# Pr2 = c[pulse2].y

w2=newmakie(lines, voltage1, Pr1)
w3=newmakie(lines, Pr1)

# w4=newmakie(lines, Voltage2, Pr2)
# w5=newmakie(lines,Pr2)

# # inlineplot() do 
# #     lines(voltage, Pr)
# # end
# # inlineplot() do 
# #     lines(Pr)
# # end
# # inlineplot() do 
# #     lines(Ex, Ey)
# # end

# figPr = Figure()
# ax = Axis(figPr[1, 1])
# lines!(ax, voltage, Pr)
# # save("D:/Code/data/shell/stripes with skymions/axayaz_1.5_1.5_1_T$(Temperature)_Amp$(Amptitude)_Speed$(Time_fctr)_80_20_20.png", figPr)
# save("D:/Code/data/shell/stripes with skymions/thickness/z=1.png", figPr)
