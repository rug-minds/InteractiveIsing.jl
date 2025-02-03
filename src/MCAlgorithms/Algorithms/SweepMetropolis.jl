struct SweepMetropolis <: MCAlgorithm end
export SweepMetropolis

function Processes.prepare(::SweepMetropolis, @specialize(args))
    (;g) = args
    return (;g,
            gstate = g.state,
            gadj = g.adj,
            gparams = g.params,
            rng = MersenneTwister(),
            lt = g[1],
            ΔH = Hamiltonian_Builder(Metropolis, g, g.hamiltonian),
            latidx = Ref(1),
            M = Ref(sum(g.state))
            )
end

@inline function SweepMetropolis(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, ΔH, lt, latidx, M) = args
    latidx[] = mod1(latidx[]+1, length(gstate))
    @inline MetropolisMag(latidx[], g, gstate, gadj, gparams, M, ΔH, lt)
end

@inline function MetropolisMag(i, g, gstate::Vector{T}, gadj, gparams, M, ΔH, lt) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[i]
    
    newstate = @inline sampleState(statetype(lt), oldstate, rng, stateset(lt))   

    ΔE = @inline ΔH(i, gstate, newstate, oldstate, gadj, gparams, lt)

    efac = exp(-β*ΔE)
    randnum = rand(rng, Float32)

    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[i] = newstate 
        M[] += (newstate - oldstate)
    end
    
    return nothing
end

struct CheckeredSweepMetropolis <: MCAlgorithm end
export CheckeredSweepMetropolis

function Processes.prepare(::CheckeredSweepMetropolis, @specialize(args))
    # Prepare two checkerboards
    (;g) = args
    first = [i for i in 1:length(g.state) if checkerboard_lat(i, size(g[1],1), false)]
    second = [i for i in 1:length(g.state) if checkerboard_lat(i, size(g[1],1), true)]
    # vcat
    checkerboards = vcat(first, second)

    (;checkerboards, prepare(SweepMetropolis(), args)...)
end

function CheckeredSweepMetropolis(@specialize(args))
    (;g, gstate, gadj, gparams, ΔH, lt, latidx, M, checkerboards) = args

    i = checkerboards[latidx[]]
    latidx[] = mod1(latidx[]+1, length(checkerboards))

    @inline MetropolisMag(i, g, gstate, gadj, gparams, M, ΔH, lt)
end

@inline function checkerboard_lat(i, size, flip = false)
    # Update by two, and shift by one every time
    # a column is finished if size is event

    col = div(i-1, size) + 1

    row = mod1(i, size)
    truth = iseven(col) == iseven(row)
    return flip ? !truth : truth
end
    