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

for g in gs # Try to not spawn more processes than CPU cores

    ## Do experiment
    ## Here we set all parameters that are used within only one experiment

    # Createprocess launches an async process, i.e. the simulation runs in the background
    # and the for loop will continue immediately, even if the simulation is not done yet
    createProcess(g, Metropolis(), lifetime = 10000)
    # DONT USE FETCH HERE Because it will halt the creation of the other processes
end

# You can fetch results Here
all_args = fetch.( gs )  # This will wait until all processes are done before returning the results
