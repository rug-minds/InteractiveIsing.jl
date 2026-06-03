export AbstractSpinGraph, AbstractIsingGraph, AbstractVectorSpinGraph
abstract type AbstractSpinGraph{T} end
abstract type AbstractIsingGraph{T} <: AbstractSpinGraph{T} end
abstract type AbstractVectorSpinGraph{T,D} <: AbstractSpinGraph{T} end
abstract type AbstractLayerData{Dim} end
abstract type AbstractIsingLayer{T,DIMS} end
abstract type AbstractLayerProperties end
abstract type StateType end


# TODO: Move these
export Discrete, Continuous, Static
struct Discrete <: StateType end
struct Continuous <: StateType end
struct Static <: StateType end
