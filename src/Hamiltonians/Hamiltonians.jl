module Hamiltonians
using ..InteractiveIsing
using ..InteractiveIsing: branchSim, setTuple

# Define hamiltonian type functions, e.g. to get params and values
include("HType.jl")

# Build the expressions for the energy functions
include("Expressions.jl")
# This means the part of the hamiltonian where the state can be factorized out
# E.g. - σ_i * H_fac (check if the minus sign is in H_fac or not)
include("Factor.jl")

# Difference term
include("Diff.jl")

# Initialize the structs as module variables
include("Vars.jl")

end