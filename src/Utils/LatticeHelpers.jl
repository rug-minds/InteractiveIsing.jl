function filter!(lattice::Lattice, f::Function)
    for i in 1:length(lattice)
        if !f(lattice[i])
            deleteat!(lattice, i)
        end
    end
end