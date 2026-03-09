function _hamiltonian_term_lines(h::Hamiltonian; io::IO = stdout)
    lines = String[]
    fnames = fieldnames(typeof(h))
    if isempty(fnames)
        push!(lines, "vars: ∅")
    else
        for name in fnames
            val = getfield(h, name)
            push!(lines, string(name, " = ", summary(val)))
        end
    end
    return lines
end

function _show_hamiltonian_terms(io::IO, hts::HamiltonianTerms)
    println(io, "HamiltonianTerms")
    hs = hamiltonians(hts)
    last_idx = length(hs)
    for (idx, h) in enumerate(hs)
        branch = idx == last_idx ? "`-- " : "|-- "
        stem = idx == last_idx ? "    " : "|   "
        println(io, branch, idx, ": ", summary(h))
        for line in _hamiltonian_term_lines(h; io)
            println(io, stem, "| ", line)
        end
    end
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", hts::HamiltonianTerms) = _show_hamiltonian_terms(io, hts)
