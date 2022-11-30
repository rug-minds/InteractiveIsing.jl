module Hamiltonians
using ..InteractiveIsing
using ..InteractiveIsing: branchSim, setTuple

# Define hamiltonian type functions, e.g. to get params and values
include("HType.jl")

# Build the expressions for the Hamiltonian factor 
# This means the part of the hamiltonian where the state can be factorized out
# E.g. - σ_i * H_fac (check if the minus sign is in H_fac or not)
include("Factor.jl")

# All other terms in the Hamiltonian
include("Term.jl")

# Initialize the structs as module variables
include("Vars.jl")

end

# list = g.adj[idx]
# sum(connW.(list).* @inbounds (@view g.state[connIdx.(list)]))