#TODO: Should be called correlation function not sampling

""" 
Takes a random number of pairs for every length to calculate correlation length function
This only works without defects.
"""
# Should all return x, y, e.g. length_bins and correlation
abstract type SamplingAlgorithm end

struct Mtl <: SamplingAlgorithm end

include("CorrelationMetal.jl")