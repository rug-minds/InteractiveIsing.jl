@ConcreteHamiltonian function Δi_H(i, gstate::Vector{T}, newstate, oldstate, gadj, gparams::Parameters, hamiltonians, layertype) where T
    # Collect the initial energy
    @initialize
    @turbo check_empty = true for ptr in nzrange(gadj, i)
        j = gadj.rowval[ptr]
        wij = gadj.nzval[ptr]

        @collect
    
    end

    return @collectsum
end

q = quote
    function Δi_H(i, gstate::Vector{T}, newstate, oldstate, gadj, gparams::Parameters, layertype) where T
        # Collect the initial energy
        @initialize
        @turbo check_empty = true for ptr in nzrange(gadj, i)
            j = gadj.rowval[ptr]
            wij = gadj.nzval[ptr]
    
            @collect
        end
        
        return @collectsum
    end
end

InteractiveIsing.replace_H_macros!(InteractiveIsing.Δi_H(), q; hamiltonians = g.hamiltonian)

println("New q: ", q)

printexp(Δi_H_exp(1, g.state, 1, 0, g.adj, g.params, g.hamiltonian, g[1]))


