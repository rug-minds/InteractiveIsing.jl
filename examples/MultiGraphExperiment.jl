using InteractiveIsing

# Initialize Graphs in a list
# Make one graph per experiment
gs = [ IsingGraph(10,10,10) for _ in 1:4]

# Define all functions and variables that will be shared across experiments
hamiltonian = Ising(g) + Quartic()
# "Broadcast" to all hamiltonians
# You can ask ChatGPT to explain broadcasting and the use of Ref
# Basically, the dot: "." applies the function to each element of the collection
# We need the Ref to avoid broadcasting over the hamiltonian itself, i.e. applying this function to all the sub-hamiltonians
# We need to use the same hamiltonian to set all graphs (if this is what we want) so we need Ref
setproperty!.(gs, Ref(:hamiltonian), Ref(hamiltonian))

for g in gs
    ## Do experiment
end