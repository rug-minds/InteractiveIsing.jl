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


createProcess(g, routine; N, sweeps, T)  

final_arguments = getargs(process(g))
total_chi = final_arguments.total_chi # Access calculated chi
println("Total Susceptibility: ", total_chi)