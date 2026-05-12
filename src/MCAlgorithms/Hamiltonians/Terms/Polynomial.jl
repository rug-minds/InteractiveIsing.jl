export PolynomialHamiltonian, Quadratic, Quartic, Sextic, Octic

"""
H = self[i]*s[i] 

The Quadratic self energy part of the Ising Hamiltonian
"""

"""
Hamiltonians of the form H = c*lp[i]*s[i]^n
"""
struct PolynomialHamiltonian{Order, P} <: LocalPotential
    parameters::P
end

PolynomialHamiltonian{Order}(params::Parameters) where Order = PolynomialHamiltonian{Order, typeof(params)}(params)

const Quadratic{P} = PolynomialHamiltonian{2, P}
const Quartic{P} = PolynomialHamiltonian{4, P}
const Sextic{P} = PolynomialHamiltonian{6, P}
const Octic{P} = PolynomialHamiltonian{8, P}

order(::Union{PolynomialHamiltonian{Order}, Type{<:PolynomialHamiltonian{Order}}}) where Order = Order

function PolynomialHamiltonian(order; c = nothing, localpotential = nothing)
    lp = localpotential
    params = Parameters(
        parameter(;
            c,
            type = AbstractArray,
            default = UniformArray(1),
            ensure = ensure_isinggraph_scalar,
            info = "Polynomial coupling constant c",
        ),
        parameter(;
            lp,
            type = AbstractArray,
            default = g -> adj(g).diag,
            default_type = Vector,
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local polynomial potential l_i",
        ),
    )
    return PolynomialHamiltonian{order, typeof(params)}(params)
end

PolynomialHamiltonian(order, c) = PolynomialHamiltonian(order; c = c)
PolynomialHamiltonian(order, c, localpotential) = PolynomialHamiltonian(order; c, localpotential)
Quadratic(;kwargs...) = PolynomialHamiltonian(2; kwargs...)
Quartic(;kwargs...) = PolynomialHamiltonian(4; kwargs...)
Sextic(;kwargs...) = PolynomialHamiltonian(6; kwargs...)
Octic(;kwargs...) = PolynomialHamiltonian(8; kwargs...)
Quadratic(params::Parameters) = PolynomialHamiltonian{2, typeof(params)}(params)
Quartic(params::Parameters) = PolynomialHamiltonian{4, typeof(params)}(params)
Sextic(params::Parameters) = PolynomialHamiltonian{6, typeof(params)}(params)
Octic(params::Parameters) = PolynomialHamiltonian{8, typeof(params)}(params)
# Quadratic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(2; c, localpotential)
# Quartic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(4; c, localpotential)
# Sextic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(6; c, localpotential)
# Octic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(8; c, localpotential)
Quadratic(c) = Quadratic(; c = c)
Quadratic(c, localpotential) = PolynomialHamiltonian(2; c, localpotential)
Quartic(c) = Quartic(; c = c)
Quartic(c, localpotential) = PolynomialHamiltonian(4; c, localpotential)
Sextic(c) = Sextic(; c = c)
Sextic(c, localpotential) = PolynomialHamiltonian(6; c, localpotential)
Octic(c) = Octic(; c = c)
Octic(c, localpotential) = PolynomialHamiltonian(8; c, localpotential)

# Quadratic(;c = ConstVal(0), localpotential = StateLike(ConstFill, 0)) = Quadratic(c, localpotential)

function instantiate(lh::PolynomialHamiltonian{Order}, g::AbstractIsingGraph) where {Order}
    params = instantiate(parameters(lh), g)
    return PolynomialHamiltonian{Order, typeof(params)}(params)
end

@inline function calculate(::ΔH, hterm::LH, model::S, proposal) where {LH <: PolynomialHamiltonian, S <: AbstractIsingGraph}
    j = at_idx(proposal)
    spins = @inline graphstate(model)
    return hterm.c[]*hterm.lp[j]*(to_val(proposal)^order(hterm) - spins[j]^order(hterm))
end

@inline function calculate(::d_iH, hterm::LH, model::S, s_idx) where {LH <: PolynomialHamiltonian, S <: AbstractIsingGraph}
    spins = @inline graphstate(model)
    return order(hterm)*hterm.c[]*hterm.lp[s_idx]*spins[s_idx]^(order(hterm)-1)
end

@inline function calculate(::H_i, hterm::LH, model::S, idx) where {LH <: PolynomialHamiltonian, S <: AbstractIsingGraph}
    spins = @inline graphstate(model)
    return hterm.c[]*hterm.lp[idx]*spins[idx]^order(hterm)
end

@inline function parameter_derivative(hterm::PolynomialHamiltonian, state::AbstractVector; dlp = similar(hterm.lp), buffermode::BufferMode = OverwriteBuffer())
    if buffermode isa OverwriteBuffer
        for i in eachindex(dlp)
            dlp[i] = hterm.c[]*state[i]^order(hterm)
        end
    else
        for i in eachindex(dlp)
            dlp[i] += sign(buffermode) * hterm.c[]*state[i]^order(hterm)
        end
    end
    return (; dlp = dlp)
end
