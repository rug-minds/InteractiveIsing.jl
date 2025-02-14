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
            M = Ref(sum(g.state)),
            lmeta = LayerMetaData(g[1])
            )
end

@inline function SweepMetropolis(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, ΔH, lmeta, latidx, M) = args
    latidx[] = mod1(latidx[]+1, length(gstate))
    @inline MetropolisMag(latidx[], g, gstate, gadj, gparams, M, ΔH, lmeta)
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
    (;g, gstate, gadj, gparams, ΔH, lmeta, latidx, M, checkerboards) = args

    i = checkerboards[latidx[]]
    latidx[] = mod1(latidx[]+1, length(checkerboards))

    @inline Metropolis(i, g, gstate, gadj, gparams, M, ΔH, lmeta)
end

@inline function checkerboard_lat(i, size, flip = false)
    # Update by two, and shift by one every time
    # a column is finished if size is event

    col = div(i-1, size) + 1

    row = mod1(i, size)
    truth = iseven(col) == iseven(row)
    return flip ? !truth : truth
end
    