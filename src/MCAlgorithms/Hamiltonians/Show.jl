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

function _compact_type_head(T::Type)
    text = string(T)
    brace_idx = findfirst(==('{'), text)
    head = isnothing(brace_idx) ? text : text[begin:prevind(text, brace_idx)]
    dot_idx = findlast(==('.'), head)
    return isnothing(dot_idx) ? head : head[nextind(head, dot_idx):end]
end

_compact_type_head(T) = _compact_type_head(typeof(T))

_hamiltonian_label(::Type{H}) where {H<:Hamiltonian} = _compact_type_head(H)
_hamiltonian_label(h::Hamiltonian) = _hamiltonian_label(typeof(h))

function Base.summary(io::IO, hterm::HamiltonianTerm)
    print(io, _hamiltonian_label(hterm))
    return nothing
end

_parameter_entry_type_summary(::Type{<:Parameter{Origin,Value}}) where {Origin<:ParameterOrigin,Value} =
    string(_compact_type_head(Value), " [", nameof(Origin), "]")

function _parameter_entry_type_summary(::Type{<:ParameterSpec{Default,Ensure,Check,Warn,Input,Units}}) where {Default,Ensure,Check,Warn,Input,Units}
    input = Input === Nothing ? "default" : _compact_type_head(Input)
    return string("spec(", input, " -> ", _compact_type_head(Default), ")")
end

_parameter_entry_type_summary(T::Type) = _compact_type_head(T)

function _show_parameters_type(io::IO, ::Type{P}; prefix = "") where {P<:Parameters}
    entries_type = P.parameters[1]
    entries_type <: NamedTuple || (print(io, prefix, _compact_type_head(P)); return nothing)

    names = fieldnames(entries_type)
    isempty(names) && (print(io, prefix, "(empty)"); return nothing)

    entry_types = fieldtypes(entries_type)
    units_type = P.parameters[3]
    last_idx = length(names)
    for (idx, name) in enumerate(names)
        branch = idx == last_idx ? "└── " : "├── "
        line_prefix = string(prefix, branch)
        entry_type = entry_types[idx]
        units = units_type <: NamedTuple && name in fieldnames(units_type) ? fieldtype(units_type, name) : Nothing
        units_text = units === Nothing ? "" : string(" {", _compact_type_head(units), "}")
        line = string(name, " = ", _parameter_entry_type_summary(entry_type), units_text)
        print(io, line_prefix, _truncate_for_prefix(io, line_prefix, line))
        idx == last_idx || println(io)
    end
    return nothing
end

function _show_hamiltonian_term_type_fields(io::IO, ::Type{H}; prefix = "") where {H<:HamiltonianTerm}
    fnames = fieldnames(H)
    isempty(fnames) && (print(io, prefix, "└── vars: ∅"); return nothing)

    field_last_idx = length(fnames)
    for (fidx, name) in enumerate(fnames)
        field_branch = fidx == field_last_idx ? "└── " : "├── "
        field_stem = fidx == field_last_idx ? "    " : "│   "
        line_prefix = string(prefix, field_branch)
        field_type = fieldtype(H, name)

        if name === :parameters && field_type <: Parameters
            println(io, line_prefix, "parameters")
            _show_parameters_type(io, field_type; prefix = string(prefix, field_stem))
        else
            line = string(name, " :: ", _compact_type_head(field_type))
            print(io, line_prefix, _truncate_for_prefix(io, line_prefix, line))
        end
        fidx == field_last_idx || println(io)
    end
    return nothing
end

function _show_hamiltonian_term_type(io::IO, ::Type{H}) where {H<:HamiltonianTerm}
    println(io, _hamiltonian_label(H))
    _show_hamiltonian_term_type_fields(io, H)
    return nothing
end

function _show_hamiltonian_terms_type(io::IO, ::Type{HTS}) where {HTS<:HamiltonianTerms}
    println(io, "HamiltonianTerms")
    HTS isa UnionAll && (print(io, "└── terms :: Tuple"); return nothing)

    hs_type = HTS.parameters[1]
    hs_type <: Tuple || (print(io, "└── ", _compact_type_head(hs_type)); return nothing)

    hs = hs_type.parameters
    isempty(hs) && (print(io, "  (empty)"); return nothing)

    last_idx = length(hs)
    for (idx, H) in enumerate(hs)
        branch = idx == last_idx ? "└── " : "├── "
        stem = idx == last_idx ? "    " : "│   "
        println(io, branch, _truncate_for_prefix(io, branch, _hamiltonian_label(H)))
        _show_hamiltonian_term_type_fields(io, H; prefix = stem)
        idx == last_idx || println(io)
    end
    return nothing
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
            field_branch = fidx == field_last_idx ? "└── " : "├── "
            field_stem = fidx == field_last_idx ? "    " : "│   "
            first_prefix = string(stem, field_branch)
            value = getfield(h, name)

            if name === :parameters && value isa Parameters
                println(io, first_prefix, "parameters")
                _show_parameters(io, value; prefix = string(stem, field_stem))
                println(io)
                continue
            end

            label = string(name, " = ")
            cols = _columns(io)
            value_cols = max(10, cols - textwidth(first_prefix) - textwidth(label) - 1)
            value_ctx = IOContext(show_ctx, :displaysize => (_rows(show_ctx), value_cols), :limit => true)
            value_text = sprint(show, value; context = value_ctx, sizehint = 0)
            line = _truncate_for_prefix(io, first_prefix, string(label, value_text))
            println(io, first_prefix, line)
        end
    end
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", hts::HamiltonianTerms) = _show_hamiltonian_terms(io, hts)
Base.show(io::IO, ::MIME"text/plain", ::Type{H}) where {H<:HamiltonianTerm} = _show_hamiltonian_term_type(io, H)
Base.show(io::IO, ::MIME"text/plain", ::Type{HTS}) where {HTS<:HamiltonianTerms} = _show_hamiltonian_terms_type(io, HTS)
