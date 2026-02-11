struct SweepMetropolis <: MCAlgorithm end
export SweepMetropolis

function Processes.init(::SweepMetropolis, @specialize(args))
    return (;latidx = Ref(1), init(Metropolis(), args)...)
end

@inline function (::SweepMetropolis)(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, ΔH, lmeta, rng, latidx, M) = args
    latidx[] = mod1(latidx[]+1, length(gstate))
    @inline Metropolis(args, latidx[], g, gstate, gadj, gparams, M, ΔH, rng, lmeta)
end


struct CheckeredSweepMetropolis <: MCAlgorithm end
export CheckeredSweepMetropolis

function ΔH_1(i, gstate, newstate, oldstate, gadj, gparams, lmeta)
    cum = zero(eltype(gstate))
    for ptr in nzrange(gadj, i)
        j = gadj.rowval[ptr]
        cum += gstate[j] 
    end
    return (oldstate-newstate) * cum
end

function Processes.init(::CheckeredSweepMetropolis, @specialize(args))
    # Init two checkerboards
    (;g) = args
    first = [i for i in 1:length(g.state) if checkerboard_lat(i, size(g[1],1), false)]
    second = [i for i in 1:length(g.state) if checkerboard_lat(i, size(g[1],1), true)]
    # vcat
    checkerboards = vcat(first, second)

    (;checkerboards, init(SweepMetropolis(), args)..., ΔH = ΔH_1)
end

function (::CheckeredSweepMetropolis)(@specialize(args))
    (;g, gstate, gadj, gparams, ΔH, lmeta, latidx, M, rng, checkerboards) = args

    i = checkerboards[latidx[]]
    latidx[] = mod1(latidx[]+1, length(checkerboards))

    @inline Metropolis(args, i, g, gstate, gadj, gparams, M, ΔH, rng, lmeta)
end

@inline function checkerboard_lat(i, size, flip = false)
    # Update by two, and shift by one every time
    # a column is finished if size is event

    col = div(i-1, size) + 1

    row = mod1(i, size)
    truth = iseven(col) == iseven(row)
    return flip ? !truth : truth
end
    