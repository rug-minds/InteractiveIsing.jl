using InteractiveIsing, GLMakie, FileIO, CairoMakie
using InteractiveIsing.StatefulAlgorithms
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
### /  \    
###     \  /
###      \/

struct TrianglePulseA{T} <: ProcessAlgorithm
    amp::T
    numpulses::Int
end
function StatefulAlgorithms.init(tp::TrianglePulseA, args)
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
function StatefulAlgorithms.step!(::TrianglePulseA, context::C) where C
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

function StatefulAlgorithms.step!(::Snapshot{DT, Name}, context::C) where {DT, Name, C}
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
function StatefulAlgorithms.init(tp::BiasA, args)
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
function StatefulAlgorithms.step!(::BiasA, context::C) where C
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
function StatefulAlgorithms.init(tp::SinPulseA, args)
    amp = tp.amp
    numpulses = tp.numpulses
    steps = num_calls(args)
    max_theta = 2*pi * numpulses

    theta = LinRange(0, max_theta, round(Int,steps))
    sins = amp .* sin.(theta)
    step = 1
    return (;sins, step, pulseval = sins[1])
end
function StatefulAlgorithms.step!(::SinPulseA, context::C) where C
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
function StatefulAlgorithms.init(tp::LinAnealingA, args)
    n_calls = num_calls(args)
    dT = (tp.stop_T - tp.start_T) / n_calls
    (;current_T = tp.start_T, dT)
end
function StatefulAlgorithms.step!(::LinAnealingA, context::C) where C
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
function StatefulAlgorithms.init(tp::LinAnealingB, args)
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
function StatefulAlgorithms.step!(::LinAnealingB, context::C) where C
    (;tem_pulse, step, model) = context

    temval = tem_pulse[step]

    temp!(model, max(temval, 0))

    return (;step = step + 1, temval)
end
##################################################################################


##################################################################################
struct ValueLogger{Name} <: ProcessAlgorithm end
ValueLogger(name) = ValueLogger{Symbol(name)}()
function StatefulAlgorithms.init(::ValueLogger, args)
    values = Float32[]
    processsizehint!(values, args)
    (;values)
end
function StatefulAlgorithms.step!(::ValueLogger, context::C) where C
    (;values, value) = context
    push!(values, value)
    return (;)
end
##################################################################################

##################################################################################
struct Recalc{I} <: StatefulAlgorithms.ProcessAlgorithm end
Recalc(i) = Recalc{Int(i)}()
function StatefulAlgorithms.step!(r::Recalc{I}, context) where I
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
function StatefulAlgorithms.init(ic::ImageCapture, input)
    (;filepath) = input
    (;callnum = 1, filepath)
end
function StatefulAlgorithms.step!(ic::ImageCapture, context::C) where C
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
function StatefulAlgorithms.init(ic::DatatoDataframe, input)
    (;filepath) = input
    (;callnum = 1, filepath)
end
dimnames(i) = (:x, :y, :z)[i]
function StatefulAlgorithms.step!(ic::DatatoDataframe, context::C) where C
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


#########################################################################





#### 可以给每一个landau项写个vector，实现不同的dipole有不同的local potential
#### StateLike可以是OffsetArray, UniformArray, ConstFill，（ConstValue = ConstFill with dimension 0）

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

#=
Spatial Landau disorder / pinned dipoles / domains:
    In the decoupled / independent localpotential form, each Landau coefficient
    can be a spatial field. The core Hamiltonian only requires
    length(localpotential) == nstates(g), so a 3D array with size
    (xL, yL, zL) can work for calculation.

    However, the new Hamiltonian viewer currently auto-detects state-sized
    parameters only when they are 1D vectors. For manual checking and
    visualization, prefer state-length vectors and build them from temporary
    3D masks with vec(mask).

    Example:
        coeff2 = fill(-2.0f0, xL * yL * zL)
        coeff4 = fill(-14.0f0, xL * yL * zL)
        coeff6 = fill(10.0f0, xL * yL * zL)
        coeff8 = fill(-1.0f0, xL * yL * zL)
        coeff10 = fill(0.2f0, xL * yL * zL)

        ## 1. weak disorder
        coeff2 .+= 0.1f0 .* randn(Float32, length(coeff2))
        coeff4 .+= 0.2f0 .* randn(Float32, length(coeff4))
        coeff6 .+= 0.1f0 .* randn(Float32, length(coeff6))
        coeff8 .+= 0.02f0 .* randn(Float32, length(coeff8))
        coeff10 .+= 0.01f0 .* randn(Float32, length(coeff10))

        ## 2. pinned region from a 3D mask
        pin = falses(xL, yL, zL)
        pin[10:18, 10:18, :] .= true
        coeff2[vec(pin)] .= 2.5f0
        coeff4[vec(pin)] .= 18.0f0
        coeff6[vec(pin)] .= 14.0f0
        coeff8[vec(pin)] .= 0.0f0
        coeff10[vec(pin)] .= 0.2f0

        ## 3. domain / layer
        layer = falses(xL, yL, zL)
        layer[:, :, div(zL, 2)] .= true
        coeff2[vec(layer)] .= 1.5f0
        coeff4[vec(layer)] .= 10.0f0
        coeff6[vec(layer)] .= 8.0f0
        coeff8[vec(layer)] .= 0.0f0
        coeff10[vec(layer)] .= 0.2f0

    Then pass these vectors as localpotential parameters:
        Ising(b = UniformArray(0.0f0), localpotential = coeff2) +
        Quartic(localpotential = coeff4) +
        Sextic(localpotential = coeff6) +
        Octic(localpotential = coeff8) +
        PolynomialHamiltonian(10; localpotential = coeff10)

    This is independent Landau; it is not the coupled mode with one global
    a,b,c shared by every dipole.
=#

## ======================== Define simulation ======================== ##
xL = 40  # Length in the x-dimension
yL = 40  # Length in the y-dimension
zL = 10   # Length in the z-dimension

### weightfunc_shell(dr,c1,c2, ax, ay, az, csr, lambda1, lambda2), Lambda is the ratio between different shells
wg1 = @WG (; dc) -> weightfunc1(; dc) NN = 3
wg2 = @WG (; dc) -> weightfunc_skymion(; dc) NN = 3
wg5 = @WG (; dc) -> weightfunc_shell(1, 1, 1, 1, 0.1, 0.1; dc) NN = 3
# Output directory for the whole sweep
outdir = raw"D:\Code\data\Manuscript\Demo1"
mkpath(outdir)
# ---- parameters to sweep ----
JIsing = 1.0
Scale = 1
Screening = 0.01 
Temp_aneal= 5f0
Temp = 0.15

a1 = -0.2
b1 = -1.4
c1 = 1
d1 = -1
e1 = 1

nspins = xL * yL * zL
coeff2 = fill(Float32(a1), nspins)
coeff4 = fill(Float32(b1), nspins)
coeff6 = fill(Float32(c1), nspins)
coeff8 = fill(Float32(d1), nspins)
coeff10 = fill(Float32(e1), nspins)

# Optional spatial disorder examples. Keep them off for the baseline check.
apply_weak_landau_disorder = false
if apply_weak_landau_disorder
    coeff2 .+= 0.1f0 .* randn(Float32, nspins)
    coeff4 .+= 0.8f0 .* randn(Float32, nspins)
    coeff6 .+= 0.5f0 .* randn(Float32, nspins)
    coeff8 .+= 0.2f0 .* randn(Float32, nspins)
    coeff10 .+= 0.2f0 .* randn(Float32, nspins)
end

proposal_delta = 0.1  # use 0.1, 0.2, 0.5 for LocalProposer(delta)
proposer_args = isnothing(proposal_delta) ? () : (LocalProposer(proposal_delta),)
g = IsingGraph(xL, yL, zL, 
        Continuous(), 
        proposer_args...,
        wg5, 
        LatticeConstants(1.0, 1.0, 1.0),
        # Ising(b = UniformArray(0), localpotential = coeff2) + 
            InteractiveIsing.MagField(b = 1) + InteractiveIsing.Bilinear() + 
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

interface(g)

# Temperature init
temp!(g, Temp)

# ----- Annealing algorithm -----
time_fctr= 1
Steps_1= 1800

Amp1 = 20
nrepeats = 2
pulse1 = TrianglePulseA(Amp1, nrepeats)
pulse2 = SinPulseA(Amp1, nrepeats)
pulse3 = Unique(SinPulseA(Amp1, nrepeats))
AnealingB = LinAnealingB(Temp_aneal, 0f0)
# algorithm_name = :metropolis
# algorithm_kwargs = (;)
# algorithm_name = :local_langevin
# algorithm_kwargs = (; stepsize = 0.02f0, adjusted = true)
# algorithm_name = :global_langevin
# algorithm_kwargs = (; stepsize = 0.02f0, adjusted = true)
algorithm_name = :block_langevin
algorithm_kwargs = (; stepsize = 0.02f0, block_size = 10, adjusted = true)
dynamics = select_dynamics(g, algorithm_name; algorithm_kwargs)


fullsweep = xL*yL*zL
point_repeat = fullsweep*time_fctr
# point_repeat = time_fctr
anneal_time = point_repeat*Steps_1
pulse_time = point_repeat*Steps_1
relax_time = point_repeat*Steps_1/2


capture_interval1 = pulse_time/(nrepeats*4)
capture_interval2 = relax_time/2 

M_Integrate_and_Logger = IntegrateAndLog(Float32, point_repeat)
B_Logger = ValueLogger(:b)
T_Logger = ValueLogger(:T)
Graph_Logger = ImageCapture(:Graph,-1.5,1.5)



# ----- Pulse Step -----
Metro_Pulse = @CompositeAlgorithm begin
    @alias dynamics = dynamics

    proposal = @every 1 dynamics()
    @every 1 M_Integrate_and_Logger(Δvalue = @transform(accepted_proposal_delta_base, proposal))
    @every point_repeat B_Logger(value = @transform(x -> x.b[], dynamics.hamiltonian))
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
    # Init(Graph_Logger, filepath = joinpath(outdir, "capture")),
    Init(M_Integrate_and_Logger, initialvalue = sum(state(g))))
c = process(g) |> fetch

# ---- Collect data2 ----
voltage2 = c[B_Logger].values
Pr2      = c[M_Integrate_and_Logger].log
# Temp1    = c[T_Logger].values
fVPr = makieaxis(f -> Axis(f[1, 1], xlabel = "Voltage", ylabel = "Pr"), ax -> lines!(ax, voltage2, Pr2))
fPr  = makieaxis(f -> Axis(f[1, 1], xlabel = "Step", ylabel = "Pr"), ax -> lines!(ax, Pr2))
