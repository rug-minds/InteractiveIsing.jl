export PolynomialHamiltonian, Quadratic, Quartic, Sextic, Octic

"""
H = self[i]*s[i] 

The Quadratic self energy part of the Ising Hamiltonian
"""

"""
Hamiltonians of the form H = c*lp[i]*s[i]^n
"""
struct PolynomialHamiltonian{Order, T, S} <: HamiltonianTerm
    c::T
    lp::S
end

PolynomialHamiltonian{Order}(c, lp) where Order = PolynomialHamiltonian{Order, typeof(c), typeof(lp)}(c, lp)

const Quadratic{T, S} = PolynomialHamiltonian{2, T, S}
const Quartic{T, S} = PolynomialHamiltonian{4, T, S}
const Sextic{T, S} = PolynomialHamiltonian{6, T, S}
const Octic{T, S} = PolynomialHamiltonian{8, T, S}

order(::Union{PolynomialHamiltonian{Order}, Type{<:PolynomialHamiltonian{Order}}}) where Order = Order

PolynomialHamiltonian(order ;c = UniformArray(1), localpotential = g -> adj(g).diag) = PolynomialHamiltonian{order}(c, localpotential)
PolynomialHamiltonian(order, c) = PolynomialHamiltonian(order; c = c)
PolynomialHamiltonian(order, c, localpotential) = PolynomialHamiltonian{order}(c, localpotential)
Quadratic(;kwargs...) = PolynomialHamiltonian(2; kwargs...)
Quartic(;kwargs...) = PolynomialHamiltonian(4; kwargs...)
Sextic(;kwargs...) = PolynomialHamiltonian(6; kwargs...)
Octic(;kwargs...) = PolynomialHamiltonian(8; kwargs...)
# Quadratic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(2; c, localpotential)
# Quartic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(4; c, localpotential)
# Sextic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(6; c, localpotential)
# Octic(;c = UniformArray(1), localpotential = StateLike(ConstFill, 0)) = PolynomialHamiltonian(8; c, localpotential)
Quadratic(c) = Quadratic(; c = c)
Quadratic(c, localpotential) = PolynomialHamiltonian{2, typeof(c), typeof(localpotential)}(c, localpotential)
Quartic(c) = Quartic(; c = c)
Quartic(c, localpotential) = PolynomialHamiltonian{4, typeof(c), typeof(localpotential)}(c, localpotential)
Sextic(c) = Sextic(; c = c)
Sextic(c, localpotential) = PolynomialHamiltonian{6, typeof(c), typeof(localpotential)}(c, localpotential)
Octic(c) = Octic(; c = c)
Octic(c, localpotential) = PolynomialHamiltonian{8, typeof(c), typeof(localpotential)}(c, localpotential)

# Quadratic(;c = ConstVal(0), localpotential = StateLike(ConstFill, 0)) = Quadratic(c, localpotential)

function reconstruct(lh::PolynomialHamiltonian, g::AbstractIsingGraph)
    T = eltype(g)
    c = map(eltype(g), lh.c)
    if lh.lp isa StateLike
        lp = lh.lp(g)
    elseif lh.lp isa Function
        lp = lh.lp(g)
    else
        lp = map(eltype(g), lh.lp)
    end
    return PolynomialHamiltonian{order(lh)}(c, lp)
end

@inline function calculate(::ΔH, hterm::LH, state::S, proposal) where {LH <: PolynomialHamiltonian, S <: AbstractIsingGraph}
    j = at_idx(proposal)
    spins = @inline graphstate(state)
    return hterm.c[]*hterm.lp[j]*(to_val(proposal)^order(hterm) - spins[j]^order(hterm))
end

@inline function calculate(::dH, hterm::LH, state::S, s_idx) where {LH <: PolynomialHamiltonian, S <: AbstractIsingGraph}
    spins = @inline graphstate(state)
    return order(hterm)*hterm.c[]*hterm.lp[s_idx]*spins[s_idx]^(order(hterm)-1)
end

@inline function calculate(::H_i, hterm::LH, state::S, idx) where {LH <: PolynomialHamiltonian, S <: AbstractIsingGraph}
    spins = @inline graphstate(state)
    return hterm.c[]*hterm.lp[idx]*spins[idx]^order(hterm)
end
