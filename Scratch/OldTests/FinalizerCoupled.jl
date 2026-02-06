abstract type AFoo end
abstract type ABar end

mutable struct Foo <: AFoo
    i::Int
    x::ABar
    t::Timer

    function Foo(i, x, t)
        f = new(i, x, t)
        finalizer(destructor, f)
        return f
    end
end

mutable struct Bar <: ABar
    i::Int
    x::AFoo
    t::Timer

    function Bar(i)
        b = new(i)

        tbar = Timer((t) -> println("bar $(b.i)"), 1, interval = 2)
        tfoo = Timer((t) -> println("foo"), 0, interval = 2)
        b.t = tbar
        b.x = Foo(i, b, tfoo)
        finalizer(destructor, b)
        return b
    end
end

function destructor(f::Foo)
    @async println("destructor foo")
    close(f.t)
end

function destructor(b::Bar)
    @async println("destructor bar")
    close(b.t)
end

b = Bar(1)
b = nothing
GC.gc(true)
