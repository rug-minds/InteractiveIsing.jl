@inline function _showctx(io::IO)
    limit = get(io, :limit, false)
    compact = get(io, :compact, false)
    color = get(io, :color, false)
    displaysize_ctx = get(io, :displaysize, displaysize(io))
    return IOContext(io, :limit => limit, :compact => compact, :color => color, :displaysize => displaysize_ctx)
end

@inline function _columns(io::IO)
    size_ctx = get(io, :displaysize, displaysize(io))
    return max(1, size_ctx[2])
end

@inline function _rows(io::IO)
    size_ctx = get(io, :displaysize, displaysize(io))
    return max(1, size_ctx[1])
end

@inline function _truncate_for_prefix(io::IO, prefix::AbstractString, text::AbstractString)
    cols = _columns(io)
    width = max(1, cols - textwidth(prefix) - 1)
    return Base._truncate_at_width_or_chars(get(io, :color, false)::Bool, text, width)
end

function _show_hamiltonian_terms(io::IO, hts::HamiltonianTerms)
    println(io, "HamiltonianTerms")
    hs = hamiltonians(hts)
    isempty(hs) && (print(io, "  (empty)"); return nothing)

    show_ctx = _showctx(io)
    last_idx = length(hs)
    for (idx, h) in enumerate(hs)
        branch = idx == last_idx ? "└── " : "├── "
        stem = idx == last_idx ? "    " : "│   "
        header_prefix = branch
        header_text = string("[", idx, "]: ", summary(h))
        println(io, header_prefix, _truncate_for_prefix(io, header_prefix, header_text))

        fnames = fieldnames(typeof(h))
        if isempty(fnames)
            println(io, stem, "└── vars: ∅")
            continue
        end

        field_last_idx = length(fnames)
        for (fidx, name) in enumerate(fnames)
            field_branch = fidx == field_last_idx ? " └── " : " ├── "
            first_prefix = string(stem, field_branch)
            label = string(name, " = ")
            cols = _columns(io)
            value_cols = max(10, cols - textwidth(first_prefix) - textwidth(label) - 1)
            value_ctx = IOContext(show_ctx, :displaysize => (_rows(show_ctx), value_cols), :limit => true)
            value_text = sprint(show, getfield(h, name); context = value_ctx, sizehint = 0)
            line = _truncate_for_prefix(io, first_prefix, string(label, value_text))
            println(io, first_prefix, line)
        end
    end
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", hts::HamiltonianTerms) = _show_hamiltonian_terms(io, hts)
