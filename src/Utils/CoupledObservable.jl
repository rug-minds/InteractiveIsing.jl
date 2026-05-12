using Observables

mutable struct CoupledObservable{T} <: AbstractObservable{T}
    listeners::Vector{Pair{Int, Any}}
    inputs::Vector{Any}  # for map!ed Observables
    ignore_equal_values::Bool
    val::T
    callbacks::Vector{Observables.MapCallback}
end

CoupledObservable(val::T) where T = CoupledObservable(Observable{T}(val), Vector{Observables.MapCallback}())
CoupledObservable{T}(val::T) where T = CoupledObservable(Observable{T}(val), Vector{Observables.MapCallback}())

function coupledmap(f, obs::AbstractObservable, args...)
    cob = CoupledObservable(f(obs.val, args...))
end