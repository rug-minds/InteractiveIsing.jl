export AcceptanceRateStepSizeTuner, DriftStepSizeTuner

"""
    AcceptanceRateStepSizeTuner(; target=0.574, gain=0.05, min_stepsize=1e-6, max_stepsize=1.0)

Adapt a routed/shared Langevin `stepsize` using the last `acceptance_rate`.

This tuner is only meaningful for `adjusted=true` Langevin algorithms. It
throws if the routed/shared `adjusted` variable is false, because unadjusted
Langevin accepts every proposal by construction.

The tuner expects its context to expose `stepsize`, `acceptance_rate`, and
`adjusted`, usually by routing or sharing those variables from a Langevin
algorithm context.
"""
struct AcceptanceRateStepSizeTuner{T<:Real} <: ProcessAlgorithm
    target::T
    gain::T
    min_stepsize::T
    max_stepsize::T
end

function AcceptanceRateStepSizeTuner(; target = 0.574, gain = 0.05, min_stepsize = 1e-6, max_stepsize = 1.0)
    target, gain, min_stepsize, max_stepsize = promote(target, gain, min_stepsize, max_stepsize)
    return AcceptanceRateStepSizeTuner(target, gain, min_stepsize, max_stepsize)
end

"""
    DriftStepSizeTuner(; target_drift=0.05, gain=0.05, min_stepsize=1e-6, max_stepsize=1.0)

Adapt a routed/shared Langevin `stepsize` so that
`stepsize * gradient_max ≈ target_drift`.

This is usable for both adjusted and unadjusted Langevin. It controls the
deterministic drift scale, not the stochastic noise scale. That makes it a
reasonable stability tuner for stiff systems, but it is not a correctness
criterion and should be treated as an interactive/warmup aid.
"""
struct DriftStepSizeTuner{T<:Real} <: ProcessAlgorithm
    target_drift::T
    gain::T
    min_stepsize::T
    max_stepsize::T
end

function DriftStepSizeTuner(; target_drift = 0.05, gain = 0.05, min_stepsize = 1e-6, max_stepsize = 1.0)
    target_drift, gain, min_stepsize, max_stepsize = promote(target_drift, gain, min_stepsize, max_stepsize)
    return DriftStepSizeTuner(target_drift, gain, min_stepsize, max_stepsize)
end

@inline _tuner_value(x) = x isa Ref ? x[] : x

@inline function _set_tuned_stepsize!(stepsize, value)
    if stepsize isa Ref
        stepsize[] = value
        return stepsize
    end
    return value
end

@inline function _bounded_multiplicative_update(current, ratio, gain, lo, hi)
    ratio = max(ratio, eps(typeof(ratio)))
    updated = current * exp(gain * log(ratio))
    return clamp(updated, lo, hi)
end

@inline function Processes.init(tuner::AcceptanceRateStepSizeTuner, context::Cont) where {Cont}
    stepsize = get(context, :stepsize, Ref(tuner.min_stepsize))
    adjusted = get(context, :adjusted, true)
    return (;stepsize, adjusted)
end

@inline function Processes.step!(tuner::AcceptanceRateStepSizeTuner, context::C) where {C}
    stepsize = context.stepsize
    current = _tuner_value(stepsize)
    acceptance_rate = _tuner_value(context.acceptance_rate)
    adjusted = _tuner_value(context.adjusted)
    adjusted || error("AcceptanceRateStepSizeTuner requires adjusted=true; unadjusted Langevin has no informative acceptance rate.")

    ratio = acceptance_rate / tuner.target
    tuned_stepsize = _bounded_multiplicative_update(current, ratio, tuner.gain, tuner.min_stepsize, tuner.max_stepsize)
    _set_tuned_stepsize!(stepsize, tuned_stepsize)
    return (;tuned_stepsize)
end

@inline function Processes.init(tuner::DriftStepSizeTuner, context::Cont) where {Cont}
    stepsize = get(context, :stepsize, Ref(tuner.min_stepsize))
    return (;stepsize)
end

@inline function Processes.step!(tuner::DriftStepSizeTuner, context::C) where {C}
    stepsize = context.stepsize
    current = _tuner_value(stepsize)
    gradient_max = max(_tuner_value(context.gradient_max), eps(typeof(current)))
    current_drift = current * gradient_max
    ratio = tuner.target_drift / current_drift
    tuned_stepsize = _bounded_multiplicative_update(current, ratio, tuner.gain, tuner.min_stepsize, tuner.max_stepsize)
    _set_tuned_stepsize!(stepsize, tuned_stepsize)
    return (;tuned_stepsize, current_drift)
end
