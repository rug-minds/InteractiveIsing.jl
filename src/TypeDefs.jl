export AbstractMonteCarloModel,
    AddOnAbstractMonteCarloModel,
    AbstractIsingGraph,
    requires,
    dependson

"""
    AbstractMonteCarloModel{T}

Common supertype for stateful objects that can be stepped by Monte Carlo
algorithms. `T` is the numeric scalar type used by the model's state and energy
calculations.
"""
abstract type AbstractMonteCarloModel{T} end

"""
    AddOnAbstractMonteCarloModel{T}

Monte Carlo model that depends on a primary model for initialization and energy
context. Add-on models can still be stepped directly by algorithms, but they
must declare the model they require with `requires` and expose the concrete
dependency with `dependson`.
"""
abstract type AddOnAbstractMonteCarloModel{T} <: AbstractMonteCarloModel{T} end

"""
    requires(model_or_type)

Return the model types required before `model_or_type` can be initialized.
Standalone Monte Carlo models require no other model by default.
"""
requires(::Type{<:AbstractMonteCarloModel}) = ()
requires(model::M) where {M<:AbstractMonteCarloModel} = requires(M)

"""
    dependson(model)

Return the concrete model dependencies held by `model`. Standalone models return
an empty tuple by default; add-on models should override this method.
"""
dependson(model::M) where {M<:AbstractMonteCarloModel} = ()

abstract type AbstractIsingGraph{T} <: AbstractMonteCarloModel{T} end
abstract type AbstractLayerData{Dim} end
abstract type AbstractIsingLayer{T,DIMS} end
abstract type AbstractLayerProperties end
abstract type StateType end


# TODO: Move these
export Discrete, Continuous, Static
struct Discrete <: StateType end
struct Continuous <: StateType end
struct Static <: StateType end
