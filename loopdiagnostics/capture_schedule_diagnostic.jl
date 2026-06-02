#!/usr/bin/env julia

using InteractiveUtils
using Printf

"""
Small mutable state box for the `step!`-style diagnostics.

The mutable variant models an algorithm whose persistent state is updated by
side effect while its diagnostic value may or may not be captured by the caller.
"""
mutable struct Box{T}
    x::T
end

"""
    step_returning(x::T, bias::T) where {T<:AbstractFloat}

Compute one dependent update and return both the new scalar state and the local
diagnostic value. This models an inlined algorithm whose return can be partially
ignored by the scheduled caller.
"""
@inline function step_returning(x::T, bias::T) where {T<:AbstractFloat}
    delta = muladd(x, T(1.000000119), bias)
    accepted = delta > T(0.125)
    newx = accepted ? x + T(0.000001) * delta : x - T(0.0000005) * delta
    return newx, delta
end

"""
    step_mutating!(box::Box{T}, bias::T) where {T<:AbstractFloat}

Compute one dependent update, write persistent state through `box`, and return
the local diagnostic value. Calls may either capture or ignore this return.
"""
@inline function step_mutating!(box::Box{T}, bias::T) where {T<:AbstractFloat}
    x = box.x
    delta = muladd(x, T(1.000000119), bias)
    accepted = delta > T(0.125)
    box.x = accepted ? x + T(0.000001) * delta : x - T(0.0000005) * delta
    return delta
end

"""
    observe(observed::T, delta::T) where {T<:AbstractFloat}

Accumulate a sampled diagnostic value so the compiler cannot delete captured
values as unused.
"""
@inline function observe(observed::T, delta::T) where {T<:AbstractFloat}
    return muladd(abs(delta), T(0.000000001), observed)
end

"""
    conditional_capture(n, period, x, bias)

Run one flat loop. Every iteration computes `delta`; only iterations satisfying
`i % period == 0` transport it into the observer accumulator.
"""
function conditional_capture(
    n::I,
    period::J,
    x::T,
    bias::T,
) where {I<:Integer,J<:Integer,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for i in one(I):n
        x, delta = @inline step_returning(x, bias)
        if rem(i, period) == 0
            observed = @inline observe(observed, delta)
        end
    end
    return x + observed
end

"""
    blocked_capture(nblocks, period, x, bias)

Run the blocked pattern: `period - 1` producer calls ignore the diagnostic
return, then one producer call captures it for the observer.
"""
function blocked_capture(
    nblocks::I,
    period::J,
    x::T,
    bias::T,
) where {I<:Integer,J<:Integer,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for _ in one(I):nblocks
        for _ in 1:(period - 1)
            x = (@inline step_returning(x, bias))[1]
        end
        x, delta = @inline step_returning(x, bias)
        observed = @inline observe(observed, delta)
    end
    return x + observed
end

"""
    conditional_capture_static(n, Val(period), x, bias)

Static-period version of `conditional_capture`. This is closer to a resolved
`CompositeAlgorithm`, where schedule intervals are carried in the type.
"""
function conditional_capture_static(
    n::I,
    ::Val{Period},
    x::T,
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for i in one(I):n
        x, delta = @inline step_returning(x, bias)
        if rem(i, Period) == 0
            observed = @inline observe(observed, delta)
        end
    end
    return x + observed
end

"""
    blocked_capture_static(nblocks, Val(period), x, bias)

Static-period version of the blocked capture pattern.
"""
function blocked_capture_static(
    nblocks::I,
    ::Val{Period},
    x::T,
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for _ in one(I):nblocks
        for _ in 1:(Period - 1)
            x = (@inline step_returning(x, bias))[1]
        end
        x, delta = @inline step_returning(x, bias)
        observed = @inline observe(observed, delta)
    end
    return x + observed
end

"""
    conditional_capture_mutating(n, period, box, bias)

Run one flat loop over a mutating `step!`-style producer. The diagnostic return
is bound every iteration but only transported on scheduled observer iterations.
"""
function conditional_capture_mutating(
    n::I,
    period::J,
    box::Box{T},
    bias::T,
) where {I<:Integer,J<:Integer,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for i in one(I):n
        delta = @inline step_mutating!(box, bias)
        if rem(i, period) == 0
            observed = @inline observe(observed, delta)
        end
    end
    return box.x + observed
end

"""
    blocked_capture_mutating(nblocks, period, box, bias)

Run the blocked mutating pattern: no-capture `step!` calls for the first
`period - 1` producer events, followed by one captured diagnostic event.
"""
function blocked_capture_mutating(
    nblocks::I,
    period::J,
    box::Box{T},
    bias::T,
) where {I<:Integer,J<:Integer,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for _ in one(I):nblocks
        for _ in 1:(period - 1)
            @inline step_mutating!(box, bias)
        end
        delta = @inline step_mutating!(box, bias)
        observed = @inline observe(observed, delta)
    end
    return box.x + observed
end

"""
    conditional_capture_mutating_static(n, Val(period), box, bias)

Static-period version of `conditional_capture_mutating`.
"""
function conditional_capture_mutating_static(
    n::I,
    ::Val{Period},
    box::Box{T},
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for i in one(I):n
        delta = @inline step_mutating!(box, bias)
        if rem(i, Period) == 0
            observed = @inline observe(observed, delta)
        end
    end
    return box.x + observed
end

"""
    blocked_capture_mutating_static(nblocks, Val(period), box, bias)

Static-period version of `blocked_capture_mutating`.
"""
function blocked_capture_mutating_static(
    nblocks::I,
    ::Val{Period},
    box::Box{T},
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    @inbounds for _ in one(I):nblocks
        for _ in 1:(Period - 1)
            @inline step_mutating!(box, bias)
        end
        delta = @inline step_mutating!(box, bias)
        observed = @inline observe(observed, delta)
    end
    return box.x + observed
end

"""
    merge_tuple_conditional(n, Val(period), x, bias)

Carry a diagnostic-like `NamedTuple` through the loop and overwrite it on every
iteration, but do not let that tuple escape. The final result uses only the
persistent scalar state plus the sampled observer accumulator.
"""
function merge_tuple_conditional(
    n::I,
    ::Val{Period},
    x::T,
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    scratch = (; delta = zero(T), accepted = false, aux = zero(T))
    @inbounds for i in one(I):n
        x, delta = @inline step_returning(x, bias)
        accepted = delta > T(0.125)
        scratch = merge(scratch, (; delta, accepted, aux = delta * T(2)))
        if rem(i, Period) == 0
            observed = @inline observe(observed, scratch.delta)
        end
    end
    return x + observed
end

"""
    merge_tuple_blocked(nblocks, Val(period), x, bias)

Blocked variant of `merge_tuple_conditional`: overwrite a non-escaping tuple on
every producer event, but only read it on the capture event.
"""
function merge_tuple_blocked(
    nblocks::I,
    ::Val{Period},
    x::T,
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    scratch = (; delta = zero(T), accepted = false, aux = zero(T))
    @inbounds for _ in one(I):nblocks
        for _ in 1:(Period - 1)
            x, delta = @inline step_returning(x, bias)
            accepted = delta > T(0.125)
            scratch = merge(scratch, (; delta, accepted, aux = delta * T(2)))
        end
        x, delta = @inline step_returning(x, bias)
        accepted = delta > T(0.125)
        scratch = merge(scratch, (; delta, accepted, aux = delta * T(2)))
        observed = @inline observe(observed, scratch.delta)
    end
    return x + observed
end

"""
    merge_tuple_dead(n, x, bias)

Overwrite a non-escaping `NamedTuple` every iteration and never read it. This
checks whether the merge is deleted entirely when it has no observable use.
"""
function merge_tuple_dead(
    n::I,
    x::T,
    bias::T,
) where {I<:Integer,T<:AbstractFloat}
    scratch = (; delta = zero(T), accepted = false, aux = zero(T))
    @inbounds for _ in one(I):n
        x, delta = @inline step_returning(x, bias)
        accepted = delta > T(0.125)
        scratch = merge(scratch, (; delta, accepted, aux = delta * T(2)))
    end
    return x
end

"""
    merge_nested_tuple_conditional(n, Val(period), x, bias)

Overwrite a nested tuple shape similar to `(; dynamics = (; delta = ...))`.
The nested tuple is loop-carried but does not escape.
"""
function merge_nested_tuple_conditional(
    n::I,
    ::Val{Period},
    x::T,
    bias::T,
) where {I<:Integer,Period,T<:AbstractFloat}
    observed = zero(T)
    scratch = (; dynamics = (; delta = zero(T), accepted = false, aux = zero(T)))
    @inbounds for i in one(I):n
        x, delta = @inline step_returning(x, bias)
        accepted = delta > T(0.125)
        scratch = merge(
            scratch,
            (; dynamics = merge(scratch.dynamics, (; delta, accepted, aux = delta * T(2)))),
        )
        if rem(i, Period) == 0
            observed = @inline observe(observed, scratch.dynamics.delta)
        end
    end
    return x + observed
end

"""
    time_call(label, f)

Warm and time a diagnostic closure, printing the final value so the whole loop
remains semantically observable.
"""
function time_call(label::AbstractString, f::F) where {F<:Function}
    result = f()
    elapsed = @elapsed result = f()
    @printf("%-32s %10.6f s    result = %.12f\n", label, elapsed, result)
    return result
end

"""
    write_llvm(path, f, types)

Write optimized LLVM for `f(types...)` to `path`.
"""
function write_llvm(path::AbstractString, f::F, types::Type) where {F}
    open(path, "w") do io
        code_llvm(io, f, types; raw = true, dump_module = false, debuginfo = :none)
    end
    return path
end

"""
    run_diagnostics()

Execute the timing comparison and write LLVM files under `loopdiagnostics/out`.
"""
function run_diagnostics()
    outdir = joinpath(@__DIR__, "out")
    mkpath(outdir)

    nblocks = 200_000
    period = 20
    n = nblocks * period
    x0 = 0.2
    bias = 0.01

    println("Timing capture schedule patterns")
    println("n = $n, period = $period, nblocks = $nblocks")
    time_call("conditional_capture", () -> conditional_capture(n, period, x0, bias))
    time_call("blocked_capture", () -> blocked_capture(nblocks, period, x0, bias))
    time_call("conditional_static", () -> conditional_capture_static(n, Val(20), x0, bias))
    time_call("blocked_static", () -> blocked_capture_static(nblocks, Val(20), x0, bias))
    time_call("conditional_mutating", () -> conditional_capture_mutating(n, period, Box(x0), bias))
    time_call("blocked_mutating", () -> blocked_capture_mutating(nblocks, period, Box(x0), bias))
    time_call("conditional_mut_static", () -> conditional_capture_mutating_static(n, Val(20), Box(x0), bias))
    time_call("blocked_mut_static", () -> blocked_capture_mutating_static(nblocks, Val(20), Box(x0), bias))
    time_call("merge_tuple_cond", () -> merge_tuple_conditional(n, Val(20), x0, bias))
    time_call("merge_tuple_blocked", () -> merge_tuple_blocked(nblocks, Val(20), x0, bias))
    time_call("merge_tuple_dead", () -> merge_tuple_dead(n, x0, bias))
    time_call("merge_nested_cond", () -> merge_nested_tuple_conditional(n, Val(20), x0, bias))

    println()
    println("Writing LLVM")
    llvm_specs = (
        ("conditional_capture.ll", conditional_capture, Tuple{Int,Int,Float64,Float64}),
        ("blocked_capture.ll", blocked_capture, Tuple{Int,Int,Float64,Float64}),
        ("conditional_static.ll", conditional_capture_static, Tuple{Int,Val{20},Float64,Float64}),
        ("blocked_static.ll", blocked_capture_static, Tuple{Int,Val{20},Float64,Float64}),
        ("conditional_mutating.ll", conditional_capture_mutating, Tuple{Int,Int,Box{Float64},Float64}),
        ("blocked_mutating.ll", blocked_capture_mutating, Tuple{Int,Int,Box{Float64},Float64}),
        ("conditional_mutating_static.ll", conditional_capture_mutating_static, Tuple{Int,Val{20},Box{Float64},Float64}),
        ("blocked_mutating_static.ll", blocked_capture_mutating_static, Tuple{Int,Val{20},Box{Float64},Float64}),
        ("merge_tuple_conditional.ll", merge_tuple_conditional, Tuple{Int,Val{20},Float64,Float64}),
        ("merge_tuple_blocked.ll", merge_tuple_blocked, Tuple{Int,Val{20},Float64,Float64}),
        ("merge_tuple_dead.ll", merge_tuple_dead, Tuple{Int,Float64,Float64}),
        ("merge_nested_tuple_conditional.ll", merge_nested_tuple_conditional, Tuple{Int,Val{20},Float64,Float64}),
    )
    for (filename, f, types) in llvm_specs
        path = joinpath(outdir, filename)
        write_llvm(path, f, types)
        println(path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_diagnostics()
end
