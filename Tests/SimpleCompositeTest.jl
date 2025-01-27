using InteractiveIsing
import InteractiveIsing as II

struct Fib end

function Fib(args)
    (;fiblist) = args
    push!(fiblist, fiblist[end] + fiblist[end-1])
end

function InteractiveIsing.prepare(::Type{Fib}, args)
    return (;fiblist = [0, 1])
end

struct Luc end

function Luc(args)
    (;luclist) = args
    push!(luclist, luclist[end] + luclist[end-1])
end

function InteractiveIsing.prepare(::Type{Luc}, args)
    return (;luclist = [2, 1])
end

FibLucComp = CompositeAlgorithm( (Fib, Luc), (1,2) )

###
struct FibLuc end

InteractiveIsing.prepare(::Type{FibLuc}, args) = (;fiblist = [0, 1], luclist = [2, 1])

function FibLuc(args)
    (;proc) = args
    Fib(args)
    if loopidx(proc) % 2 == 0
        Luc(args)
    end
end

struct FibLucTrig{Intervals} end

function InteractiveIsing.prepare(::Type{FibLucTrig{Intervals}}, args) where Intervals
    (;runtime) = args
    rpts = InteractiveIsing.repeats(runtime)
    triggers = ((InteractiveIsing.InitTriggerList(interval) for interval in Intervals)...,)
    for i in 1:rpts
        for (i_idx, interval) in enumerate(Intervals)
            if i % interval == 0
                push!(triggers[i_idx].triggers, i)
            end
        end
    end
    triggers = InteractiveIsing.CompositeTriggers(triggers)
    return (;fiblist = [0, 1], luclist = [2, 1], triggers)
end

function FibLucTrig{(1,2)}(args)
    (;proc, triggers) = args
    Fib(args)
    InteractiveIsing.skiplist!(triggers)
    if InteractiveIsing.shouldtrigger(triggers, loopidx(proc))
        Luc(args)
        InteractiveIsing.inc!(triggers)
    end
    InteractiveIsing.skiplist!(triggers)
end

benchmark(FibLucComp, 1000000)
benchmark(FibLuc, 1000000)
benchmark(FibLucTrig{(1,2)}, 1000000)

benchmark(FibLucComp, 1000000, loopfunction = unrollloop)

# p = InteractiveIsing.Process(FibLucComp, 100)

# p1 = InteractiveIsing.Process(FibLucComp, 100)
# p2 = InteractiveIsing.Process(FibLuc, 100)
# p3 = InteractiveIsing.Process(FibLucTrig{(1,2)}, 100)

# process_warntype(p1)

# process_warntype(p2)
# process_warntype(p3)

# start(p1)
# start(p2)
# start(p3)


# start(p)

