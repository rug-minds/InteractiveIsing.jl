struct ExampleHamiltonian{T} <: Hamiltonian
    value::T
end

# Either define the Hamiltonian in index notation 
# => System will auto generate difference based on FlipProposal
ΔH_expr[ExampleHamiltonian] = :(s[i]*self[i]*value[])

# Or define explicit energy difference
function ΔH(ch::ExampleHamiltonian, hargs, delta)
    ch.value
end

# Define how the Hamiltonian updates internal state
# At the end of the Metropolis step
# May use the args that are prepared for the Metropolis algorithm
function update!(ch::ExampleHamiltonian, args)
    ch.value += 1
    return ch
end

