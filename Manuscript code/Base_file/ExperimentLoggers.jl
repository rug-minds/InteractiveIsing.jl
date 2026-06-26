################################################################################
# Runtime loggers and their energy diagnostics
################################################################################

struct ValueLogger{Name} <: ProcessAlgorithm end

ValueLogger(name) = ValueLogger{Symbol(name)}()

function StatefulAlgorithms.init(::ValueLogger, args)
    values = Float32[]
    processsizehint!(values, args)
    return (; values)
end

function StatefulAlgorithms.step!(::ValueLogger, context::C) where C
    (; values, value) = context
    push!(values, value)
    return (;)
end

################################################################################

struct DepolLogger{Name} <: ProcessAlgorithm end

DepolLogger(name) = DepolLogger{Symbol(name)}()

function StatefulAlgorithms.init(::DepolLogger, args)
    means = Float32[]
    medians = Float32[]
    maxima = Float32[]
    total_energy = Float32[]
    depol_energy = Float32[]
    interaction_energy = Float32[]
    field_energy = Float32[]
    poly_energy = Float32[]

    processsizehint!(means, args)
    processsizehint!(medians, args)
    processsizehint!(maxima, args)
    processsizehint!(total_energy, args)
    processsizehint!(depol_energy, args)
    processsizehint!(interaction_energy, args)
    processsizehint!(field_energy, args)
    processsizehint!(poly_energy, args)

    return (;
        means,
        medians,
        maxima,
        total_energy,
        depol_energy,
        interaction_energy,
        field_energy,
        poly_energy,
    )
end

function StatefulAlgorithms.step!(::DepolLogger, context::C) where C
    (;
        means,
        medians,
        maxima,
        total_energy,
        depol_energy,
        interaction_energy,
        field_energy,
        poly_energy,
        model,
        hamiltonian,
    ) = context

    depol_term = find_first_term(hamiltonian, InteractiveIsing.CoulombHamiltonian)
    push!(total_energy, total_supported_energy(hamiltonian, model))
    push!(interaction_energy, bilinear_total_energy(hamiltonian, model))
    push!(field_energy, sum_term_energy_by_type(hamiltonian, model, InteractiveIsing.ExtField))
    push!(poly_energy, total_polynomial_energy(hamiltonian, model))

    isnothing(depol_term) && return (;)

    depol = coulomb_local_scale(model, depol_term)
    push!(depol_energy, coulomb_total_energy(depol_term))
    push!(means, Float32(depol.mean))
    push!(medians, Float32(depol.median))
    push!(maxima, Float32(depol.maximum))
    return (;)
end

################################################################################

struct IntegrateAndLog{T} <: ProcessAlgorithm
    loginterval::Int
end

IntegrateAndLog(type = Float64, loginterval = 1) =
    IntegrateAndLog{type}(Int(loginterval))

function StatefulAlgorithms.init(logger::IntegrateAndLog{T}, args) where T
    total = convert(T, get(args, :initialvalue, zero(T)))
    log = T[]
    processsizehint!(log, args)
    return (; total, log, step = 0)
end

function StatefulAlgorithms.step!(logger::IntegrateAndLog{T}, context) where T
    (; total, log, step, value) = context
    total += convert(T, value)
    step += 1
    if step % logger.loginterval == 0
        push!(log, total)
    end
    return (; total, step)
end

################################################################################
# Hamiltonian diagnostics used by DepolLogger and reduced-parameter reporting
################################################################################

function find_first_term(hts, ::Type{T}) where T
    for h in InteractiveIsing.hamiltonians(hts)
        h isa T && return h
    end
    return nothing
end

function sum_term_energy_by_type(hamiltonian, model, ::Type{T}) where T
    total = 0.0f0
    for h in InteractiveIsing.hamiltonians(hamiltonian)
        if h isa T
            total += Float32(InteractiveIsing.calculate(InteractiveIsing.H(), h, model))
        end
    end
    return total
end

function bilinear_total_energy(hamiltonian, model)
    s = state(model)
    total = 0.0f0

    for h in InteractiveIsing.hamiltonians(hamiltonian)
        if h isa InteractiveIsing.Bilinear
            local_total = 0.0f0
            for i in eachindex(s)
                proposal = SingleSpinProposal(i, s[i], NoChange(), 1)
                local_total += Float32(s[i]) * Float32(
                    InteractiveIsing.calculate(InteractiveIsing.d_iH(), h, model, proposal),
                )
            end
            total += 0.5f0 * local_total
        end
    end

    return total
end

function total_polynomial_energy(hamiltonian, model)
    total = 0.0f0
    s = state(model)

    for h in InteractiveIsing.hamiltonians(hamiltonian)
        if h isa InteractiveIsing.PolynomialHamiltonian
            local_total = 0.0f0
            for i in eachindex(s)
                local_total += Float32(
                    InteractiveIsing.calculate(InteractiveIsing.H_i(), h, model, i),
                )
            end
            total += local_total
        end
    end

    return total
end

function coulomb_total_energy(hamiltonian)
    depol_term = hamiltonian isa InteractiveIsing.CoulombHamiltonian ?
        hamiltonian :
        find_first_term(hamiltonian, InteractiveIsing.CoulombHamiltonian)

    isnothing(depol_term) && return 0.0f0
    return 0.5f0 * Float32(sum(depol_term.ρ .* depol_term.u))
end

function total_supported_energy(hamiltonian, model)
    total = 0.0f0
    total += bilinear_total_energy(hamiltonian, model)
    total += sum_term_energy_by_type(hamiltonian, model, InteractiveIsing.ExtField)
    total += total_polynomial_energy(hamiltonian, model)
    total += coulomb_total_energy(hamiltonian)
    return total
end

function coulomb_local_scale(g, coulomb_term)
    vals = Float64[]
    for i in eachindex(state(g))
        proposal = SingleSpinProposal(i, state(g)[i], NoChange(), 1)
        local_field = InteractiveIsing.calculate(
            InteractiveIsing.d_iH(),
            coulomb_term,
            g,
            proposal,
        )
        push!(vals, abs(local_field))
    end
    return (; mean = mean(vals), median = median(vals), maximum = maximum(vals))
end
