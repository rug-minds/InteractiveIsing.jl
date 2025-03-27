using InteractiveIsing, Processes

struct CalcSusceptibility <: ProcessAlgorithm end

function incs_per_sec(p::Process)
    loopidx(p) / runtime(p)
end

# Runs one time at the beginning
function Processes.prepare(::CalcSusceptibility, args)
    M1s = Float64[]  # Store normalized magnetization for each sweep
    M2s = Float64[]  # Store squared normalized magnetization for each sweep
    processsizehint!(args, M1s) 
    processsizehint!(args, M2s) 
    return (;M1s, M2s)
end

# Once per simulation step
function (::CalcSusceptibility)(args)
    (;g, gstate, M, M1s, M2s, N) = args
    
    Mag_normalized = M[] / (N^2)  
    push!(M1s, Mag_normalized)  # Store normalized magnetization
    push!(M2s, Mag_normalized^2)  # Store squared normalized magnetization
    
    return
end

# Runs one time at the end
function Processes.cleanup(::CalcSusceptibility, args)
    (;M1s, M2s, g, sweeps, N, T) = args 

    T = T # Get temperature
    

    accumulated_M1 = sum(M1s)  # Accumulated normalized magnetization
    accumulated_M2 = sum(M2s)  # Accumulated squared normalized magnetization

    avg_M = accumulated_M1 / sweeps 
    avg_M2 = accumulated_M2 / sweeps 
    total_chi = (1.0 / T) * ((avg_M2 - avg_M^2) * (N^2)) 

    return (;total_chi) # Return calculated chi
end

N = 64
wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
g = IsingGraph(N,N, type = Discrete, weights = wg)
#interface(g)
T=2.269
eqsteps = 40
temp(g, 2.27)
sweeps = 10000

onesweep = N^2

Equilibration = CheckeredSweepMetropolis
SweepSusceptibility = CompositeAlgorithm( (CheckeredSweepMetropolis, CalcSusceptibility), (1, onesweep))

routine = Routine((Equilibration, SweepSusceptibility), (eqsteps*onesweep, sweeps*onesweep))




# final_arguments = getargs(process(g))
total_chi = final_arguments.total_chi # Access calculated chi
println("Total Susceptibility: ", total_chi)
using Random
mutable struct SimpleRNG{T} <: Random.AbstractRNG
    state::T
    const stepsize::T
    const lowval::T
    const highval::T
end

Base.length(rng::SimpleRNG) = rng.highval - rng.lowval

function Random.rand(rng::SimpleRNG{T}, ::Type{T} = T) where T
    rng.state = mod(rng.state + rng.stepsize + rng.lowval, rng.highval) - rng.lowval
    return rng.state
end

rng = SimpleRNG(0.2f0, 10*Float32(pi), 0f0, 1f0)

createProcess(g, SweepMetroplis, overrides = (;rng))
# function Random.rand(rng::SimpleRNG{T}, ar::AbstractRange) where Task
#     rangel = length(ar)
#     #remap idx onto the range of the range
#     idx = 
# end

# createProcess(g, routine; N, sweeps, T, overrides = (;rng)) 


