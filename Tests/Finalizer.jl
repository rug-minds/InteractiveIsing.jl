mutable struct MyTimer
    timer::Timer
    function MyTimer(timer)
        mt = new(timer)
        finalizer(destructor, mt)
        return mt
    end
end

function destructor(mt)
    close(mt.timer)
end

mytimer = MyTimer(Timer((t) -> println("test $t"), 0, interval = 1))

# mytimer = nothing
# GC.gc()