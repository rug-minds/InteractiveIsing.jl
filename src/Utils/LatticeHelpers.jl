function filter!(lattice::Lattice, f::Function)
    for i in eachindex(lattice)
        if !f(lattice[i])
            deleteat!(lattice, i)
        end
    end
end