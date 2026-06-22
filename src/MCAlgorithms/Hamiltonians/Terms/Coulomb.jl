# TODO: We need a conversion factor from dipole to charge
export CoulombHamiltonian, init!, precompute_du_self!, recalc!, recalcnew, ΔH, update!,
       PositiveFreeCharge, NegativeFreeCharge, free_charge_total,
       validate_free_charge_neutrality!, rebuild_charge_density!,
       add_cell_free_charge!, remove_cell_free_charge!, move_cell_free_charge!,
       add_sheet_free_charge!, remove_sheet_free_charge!, move_sheet_free_charge!

"""
We have a charge field where s_i -> (q_i+1, q_i) for the

The energy becomes H = 1/2 ϕ_i q_i
"""
# Template rewrite note:
# The old CoulombHamiltonian stored parameters, buffers, FFT plans, Thomas
# factors, and recalc state as one flat struct. That flat storage is intentionally
# replaced here by:
#
#     CoulombHamiltonian(parameters, internal)
#
# where `parameters` contains symbolic/template Hamiltonian parameters such as
# the scaling factor, while `CoulombInternal` contains scratch buffers, plans,
# lattice constants, boundary configuration such as screening lengths, and
# recalc state. The old numerical code below is kept active by generic property
# forwarding (`c.ρ`, `c.scaling`, `c.recalc_steps`, etc.), but the old flat
# struct layout should not be restored.
abstract type AbstractFreeChargeSign end

"""
    PositiveFreeCharge()

Marker for positive mobile free-charge occupancy stored by `CoulombHamiltonian`.
"""
struct PositiveFreeCharge <: AbstractFreeChargeSign end

"""
    NegativeFreeCharge()

Marker for negative mobile free-charge occupancy stored by `CoulombHamiltonian`.
"""
struct NegativeFreeCharge <: AbstractFreeChargeSign end

struct CoulombInternal{T,PxyT,IPxyT,N,PO,NO,PS,NS,R} <: InternalImplementation
    size::NTuple{N,Int}
    ρ::Array{T,3}                 # real charges
    ρhat::Array{Complex{T},3}     # rfft(ρ) over (x,y)
    uhat::Array{Complex{T},3}     # spectral potential (same size as ρhat)
    u::Array{T,3}                 # real potential
    screen_top::T
    screen_bot::T
    ax::T
    ay::T
    az::T
    ϵ::T
    du_self::Vector{T}            # per-layer unit charge-pair field jump du[z+1]-du[z]

    Pxy::PxyT                     # batched plan_rfft over x/y for all z planes
    iPxy::IPxyT                   # batched unnormalized plan_brfft over x/y for all z planes

    mod_upperd::Array{T,3}        # Thomas algorithm modified upper diagonal (cp), size (Nxh, Ny, Nz)
    inv_den::Array{T,3}
    dp_scratch::Array{Complex{T},3} # Thomas algorithm forward sweep scratch space

    positive_cell_occupancy::PO    # conserved positive free-charge counts on dipole cells
    negative_cell_occupancy::NO    # conserved negative free-charge counts on dipole cells
    positive_sheet_occupancy::PS   # conserved positive free-charge counts on charge sheets
    negative_sheet_occupancy::NS   # conserved negative free-charge counts on charge sheets
    q_positive::T
    q_negative::T
    free_charge_split::T

    recalc_steps::R
    recalc_tracker::Base.RefValue{Int} # Counter to track when to recalculate potential (for external coupling)
end

struct CoulombHamiltonian{P,I} <: LayerTerm
    layer::Int
    parameters::P
    internal::I
end

Base.size(c::CoulombHamiltonian) = c.size

"""
    _coulomb_recalc_steps(recalc)

Normalize a user recalc cadence. `Inf` disables scheduled spin-update recalc;
finite values are positive integer update counts.
"""
function _coulomb_recalc_steps(recalc)
    isinf(recalc) && return Inf
    steps = Int(recalc)
    steps > 0 || throw(ArgumentError("CoulombHamiltonian recalc must be positive or Inf; got $recalc."))
    steps == recalc || throw(ArgumentError("CoulombHamiltonian recalc must be an integer cadence or Inf; got $recalc."))
    return steps
end

"""
    _coulomb_recalc_tracker(offset, recalc_steps)

Return the initial scheduled-recalc tracker after applying `offset`.
"""
function _coulomb_recalc_tracker(offset, recalc_steps)
    isinf(recalc_steps) && return 1
    offset_int = Int(offset)
    offset_int == offset || throw(ArgumentError("CoulombHamiltonian recalc_offset must be an integer; got $offset."))
    return mod1(offset_int + 1, Int(recalc_steps))
end

@inline function CoulombHamiltonian(;
    layer = 1,
    scaling = 1.f0,
    screening = Inf32,
    screen_len_top = screening,
    screen_len_bot = screening,
    recalc = 1,
    recalc_offset = 0,
    q_positive = 1.f0,
    q_negative = 1.f0,
    free_charge_split = 0.5f0,
)
    params = Parameters(
        parameter(;
            scaling,
            type = AbstractArray,
            default = ConstVal(1f0),
            ensure = ensure_isinggraph_scalar,
            info = "Dipole-to-charge scaling factor",
            units = physicalunits(dipole = 1, role = :dipole_scale),
        ),
    )
    internal = InternalPlan((; screen_len_top, screen_len_bot, recalc, recalc_offset, q_positive, q_negative, free_charge_split)) do plan, g
        config = plan.values
        T = eltype(g)
        Nx, Ny, Nz_dip = size(g)
        Nz = Nz_dip + 1
        Nxh = Nx ÷ 2 + 1

        charge_size = (Nx, Ny, Nz)
        spectral_size = (Nxh, Ny, Nz)
        constants = lattice_constants(top(g))

        ρ = zeros(T, charge_size)
        ρhat = zeros(Complex{T}, spectral_size)
        uhat = zeros(Complex{T}, spectral_size)
        u = zeros(T, charge_size)
        positive_cell_occupancy = zeros(Int, Nx, Ny, Nz_dip)
        negative_cell_occupancy = zeros(Int, Nx, Ny, Nz_dip)
        positive_sheet_occupancy = zeros(Int, charge_size)
        negative_sheet_occupancy = zeros(Int, charge_size)
        recalc_steps = _coulomb_recalc_steps(config.recalc)
        recalc_tracker = Ref(_coulomb_recalc_tracker(config.recalc_offset, recalc_steps))

        return CoulombInternal(
            charge_size,                                  # size
            ρ,                                            # ρ
            ρhat,                                         # ρhat
            uhat,                                         # uhat
            u,                                            # u
            T(internalvalue(config.screen_len_top, physicalunits(length = 1), physicalscales(g), g; parameter = :screen_len_top)), # screen_top
            T(internalvalue(config.screen_len_bot, physicalunits(length = 1), physicalscales(g), g; parameter = :screen_len_bot)), # screen_bot
            T(constants[1]),                              # ax
            T(constants[2]),                              # ay
            T(constants[3]),                              # az
            one(T),                                       # ϵ
            zeros(T, Nz_dip),                             # du_self
            plan_rfft(ρ, (1, 2); flags = FFTW.MEASURE), # Pxy
            plan_brfft(uhat, Nx, (1, 2); flags = FFTW.MEASURE), # iPxy
            zeros(T, spectral_size),                      # mod_upperd
            zeros(T, spectral_size),                      # inv_den
            zeros(Complex{T}, spectral_size),             # dp_scratch
            positive_cell_occupancy,                      # positive_cell_occupancy
            negative_cell_occupancy,                      # negative_cell_occupancy
            positive_sheet_occupancy,                     # positive_sheet_occupancy
            negative_sheet_occupancy,                     # negative_sheet_occupancy
            T(internalvalue(config.q_positive, physicalunits(charge = 1), physicalscales(g), g; parameter = :q_positive)), # q_positive
            T(internalvalue(config.q_negative, physicalunits(charge = 1), physicalscales(g), g; parameter = :q_negative)), # q_negative
            T(config.free_charge_split),                  # free_charge_split
            recalc_steps,                                 # recalc_steps
            recalc_tracker,                               # recalc_tracker
        )
    end
    return CoulombHamiltonian(Int(layer), params, internal)
end

@inline CoulombHamiltonian(layer::Integer; kwargs...) = CoulombHamiltonian(; layer, kwargs...)

@inline function CoulombHamiltonian(
    g::AbstractIsingGraph,
    scaling = 1.f0;
    layer = 1,
    screening = Inf32,
    screen_len_top = screening,
    screen_len_bot = screening,
    recalc = 1,
    recalc_offset = 0,
    q_positive = 1.f0,
    q_negative = 1.f0,
    free_charge_split = 0.5f0,
)
    h = instantiate(
        CoulombHamiltonian(;
            layer,
            scaling,
            screening,
            screen_len_top,
            screen_len_bot,
            recalc,
            recalc_offset,
            q_positive,
            q_negative,
            free_charge_split,
        ),
        g,
    )
    return init!(h, g)
end

function instantiate(c::CoulombHamiltonian, g::AbstractIsingGraph)
    # `scaling` is stored here as the raw user parameter. Charge conversion is
    # applied later as `c.scaling[] / c.az`, so re-instantiation must rebuild
    # buffers for the new lattice but must not renormalize `scaling` itself.
    layer = boundlayer(c, g)
    h = CoulombHamiltonian(
        layeridx(c),
        instantiate(parameters(c), layer),
        instantiate(internal(c), layer),
    )
    return init!(h, layer)
end

function init!(c::CoulombHamiltonian, g::AbstractIsingGraph)
    return init!(c, boundlayer(c, g))
end

function init!(c::CoulombHamiltonian, layer::AbstractIsingLayer)
    rebuild_charge_density!(c, layer)
    precompute_solve_factors!(c)
    precompute_du_self!(c)
    recalc!(c)

    return c
end

"""
    rebuild_charge_density!(coulomb, layer; validate=true)

Rebuild the total sheet charge `ρ` from bound dipoles plus the conserved
positive and negative free-charge occupancies stored on `coulomb`.
"""
function rebuild_charge_density!(c::C, layer::L; validate::Bool = true) where {C<:CoulombHamiltonian,L<:AbstractIsingLayer}
    validate && validate_free_charge_neutrality!(c)

    ρ = c.ρ
    spins = graphstate(layer)

    Nx, Ny, Nz = size(ρ)
    Nz_dip = Nz - 1
    # Convert the stored dipole scaling to charge scaling for this lattice.
    scaling = c.scaling[] / c.az

    # zero charge buffers
    fill!(ρ, zero(eltype(ρ)))

    # accumulate bound charges from dipoles
    @inbounds for z in 1:Nz_dip
        @inbounds for j in 1:Ny, i in 1:Nx
            v = spins[i, j, z]
            v = v * scaling # Scaling factor dipole to charge
            ρ[i,j,z]   -= v
            ρ[i,j,z+1] += v
        end
    end
    _add_free_charge_density!(c)
    return c
end

"""
    free_charge_total(coulomb)

Return total mobile free charge from positive and negative cell/sheet
occupancies. Neutral free-charge configurations return zero.
"""
function free_charge_total(c::C) where {C<:CoulombHamiltonian}
    T = eltype(c.ρ)
    npositive = sum(c.positive_cell_occupancy) + sum(c.positive_sheet_occupancy)
    nnegative = sum(c.negative_cell_occupancy) + sum(c.negative_sheet_occupancy)
    return T(c.q_positive) * T(npositive) - T(c.q_negative) * T(nnegative)
end

"""
    validate_free_charge_neutrality!(coulomb; atol=sqrt(eps(T)))

Throw an `ArgumentError` unless the stored positive and negative free-charge
occupancies are neutral after applying their charge magnitudes.
"""
function validate_free_charge_neutrality!(c::C; atol = sqrt(eps(eltype(c.ρ)))) where {C<:CoulombHamiltonian}
    total = free_charge_total(c)
    isapprox(total, zero(total); atol) && return c
    throw(ArgumentError("CoulombHamiltonian free charge must be neutral; total free charge is $total. Add explicit opposite charge occupancy before rebuilding the Coulomb field."))
end

@inline _free_cell_occupancy(c::CoulombHamiltonian, ::PositiveFreeCharge) = c.positive_cell_occupancy
@inline _free_cell_occupancy(c::CoulombHamiltonian, ::NegativeFreeCharge) = c.negative_cell_occupancy
@inline _free_sheet_occupancy(c::CoulombHamiltonian, ::PositiveFreeCharge) = c.positive_sheet_occupancy
@inline _free_sheet_occupancy(c::CoulombHamiltonian, ::NegativeFreeCharge) = c.negative_sheet_occupancy

@inline _free_charge_magnitude(c::CoulombHamiltonian, ::PositiveFreeCharge) = c.q_positive
@inline _free_charge_magnitude(c::CoulombHamiltonian, ::NegativeFreeCharge) = c.q_negative
@inline _free_charge_sign(::PositiveFreeCharge) = 1
@inline _free_charge_sign(::NegativeFreeCharge) = -1

"""
    add_cell_free_charge!(coulomb, sign, coord[, count])

Add `count` mobile free charges of `sign` to a dipole-cell coordinate. This
updates occupancy only; call `rebuild_charge_density!` before solving.
"""
function add_cell_free_charge!(c::C, sign::S, coord::CartesianIndex{3}, count::I = 1) where {C<:CoulombHamiltonian,S<:AbstractFreeChargeSign,I<:Integer}
    count >= 0 || throw(ArgumentError("Free-charge occupancy count must be nonnegative; got $count."))
    occupancy = _free_cell_occupancy(c, sign)
    checkbounds(occupancy, coord)
    @inbounds occupancy[coord] += Int(count)
    return c
end

"""
    remove_cell_free_charge!(coulomb, sign, coord[, count])

Remove `count` mobile free charges of `sign` from a dipole-cell coordinate.
This updates occupancy only; call `rebuild_charge_density!` before solving.
"""
function remove_cell_free_charge!(c::C, sign::S, coord::CartesianIndex{3}, count::I = 1) where {C<:CoulombHamiltonian,S<:AbstractFreeChargeSign,I<:Integer}
    count >= 0 || throw(ArgumentError("Free-charge occupancy count must be nonnegative; got $count."))
    occupancy = _free_cell_occupancy(c, sign)
    checkbounds(occupancy, coord)
    @inbounds current = occupancy[coord]
    current >= count || throw(ArgumentError("Cannot remove $count free charges from $coord; only $current are present."))
    @inbounds occupancy[coord] = current - Int(count)
    return c
end

"""
    move_cell_free_charge!(coulomb, sign, from, to[, count])

Move `count` mobile free charges of `sign` between two dipole cells. This
preserves total free charge by construction.
"""
function move_cell_free_charge!(c::C, sign::S, from::CartesianIndex{3}, to::CartesianIndex{3}, count::I = 1) where {C<:CoulombHamiltonian,S<:AbstractFreeChargeSign,I<:Integer}
    remove_cell_free_charge!(c, sign, from, count)
    add_cell_free_charge!(c, sign, to, count)
    return c
end

"""
    add_sheet_free_charge!(coulomb, sign, coord[, count])

Add `count` direct sheet free charges of `sign` to a Coulomb charge-sheet
coordinate. This updates occupancy only; call `rebuild_charge_density!` before
solving.
"""
function add_sheet_free_charge!(c::C, sign::S, coord::CartesianIndex{3}, count::I = 1) where {C<:CoulombHamiltonian,S<:AbstractFreeChargeSign,I<:Integer}
    count >= 0 || throw(ArgumentError("Free-charge occupancy count must be nonnegative; got $count."))
    occupancy = _free_sheet_occupancy(c, sign)
    checkbounds(occupancy, coord)
    @inbounds occupancy[coord] += Int(count)
    return c
end

"""
    remove_sheet_free_charge!(coulomb, sign, coord[, count])

Remove `count` direct sheet free charges of `sign` from a Coulomb charge-sheet
coordinate. This updates occupancy only; call `rebuild_charge_density!` before
solving.
"""
function remove_sheet_free_charge!(c::C, sign::S, coord::CartesianIndex{3}, count::I = 1) where {C<:CoulombHamiltonian,S<:AbstractFreeChargeSign,I<:Integer}
    count >= 0 || throw(ArgumentError("Free-charge occupancy count must be nonnegative; got $count."))
    occupancy = _free_sheet_occupancy(c, sign)
    checkbounds(occupancy, coord)
    @inbounds current = occupancy[coord]
    current >= count || throw(ArgumentError("Cannot remove $count free charges from $coord; only $current are present."))
    @inbounds occupancy[coord] = current - Int(count)
    return c
end

"""
    move_sheet_free_charge!(coulomb, sign, from, to[, count])

Move `count` direct sheet free charges of `sign` between two Coulomb charge
sheets. This preserves total free charge by construction.
"""
function move_sheet_free_charge!(c::C, sign::S, from::CartesianIndex{3}, to::CartesianIndex{3}, count::I = 1) where {C<:CoulombHamiltonian,S<:AbstractFreeChargeSign,I<:Integer}
    remove_sheet_free_charge!(c, sign, from, count)
    add_sheet_free_charge!(c, sign, to, count)
    return c
end

"""
    _add_free_charge_density!(coulomb)

Project cell-centered mobile free charge and direct sheet free charge into the
derived total sheet charge `ρ`.
"""
function _add_free_charge_density!(c::C) where {C<:CoulombHamiltonian}
    ρ = c.ρ
    T = eltype(ρ)
    split = T(c.free_charge_split)
    zero(T) <= split <= one(T) ||
        throw(ArgumentError("CoulombHamiltonian free_charge_split must lie in [0, 1]; got $(c.free_charge_split)."))

    qpos = T(c.q_positive)
    qneg = T(c.q_negative)
    pos_cell = c.positive_cell_occupancy
    neg_cell = c.negative_cell_occupancy
    Nx, Ny, Nz_dip = size(pos_cell)

    @inbounds for z in 1:Nz_dip, j in 1:Ny, i in 1:Nx
        q = qpos * T(pos_cell[i, j, z]) - qneg * T(neg_cell[i, j, z])
        ρ[i, j, z] += q * (one(T) - split)
        ρ[i, j, z + 1] += q * split
    end

    @inbounds for idx in eachindex(ρ)
        ρ[idx] += qpos * T(c.positive_sheet_occupancy[idx]) - qneg * T(c.negative_sheet_occupancy[idx])
    end
    return c
end


function compute_ktilde2(c::CoulombHamiltonian, nx, ny)
    T = eltype(c.ρ)
    ax = c.ax
    ay = c.ay

    Nx = c.size[1]
    Ny = c.size[2]
    twopi = T(2) * T(π)
    # Map FFT indices to signed mode indices: 0,1,...,N/2, -(N/2-1),..., -1
    mx = nx - 1 # FFT index to mode index
    my = (ny-1 <= Ny÷2) ? (ny-1) : (ny-1 - Ny)

    kx = twopi * mx / (Nx * ax)
    ky = twopi * my / (Ny * ay)

    kxt = 2 * sin(0.5 * kx * ax) / ax
    kyt = 2 * sin(0.5 * ky * ay) / ay
    return kxt^2 + kyt^2
end

"""
Precompute Thomas-factorization-like arrays for each (kx,ky) mode.

    We store:
    invden[kx,ky,n] = 1 / den_n
    m_upper[kx,ky,n] = c'_n  (modified upper); m_upper[:,:,Nz] = 0

    System per mode:
    (α_bot - c) ϕ̂₁ +      ϕ̂₂           = ŝ₁
        ϕ̂ₙ₋₁  - c ϕ̂ₙ + ϕ̂ₙ₊₁         = ŝₙ   (2..Nz-1)
        ϕ̂Nz-1 + (α_top - c) ϕ̂Nz         = ŝNz

    where c = 2 + k̃² * az², α_bot = 1 - az/λbot, α_top = 1 - az/λtop.
"""
function precompute_solve_factors!(ch::CoulombHamiltonian)
    T = eltype(ch.ρ)
    m_upper = ch.mod_upperd #modified upper diagonal,
    invden = ch.inv_den
    Nx, _, _ = size(ch.ρ)
    Nxh, Ny, Nz = size(invden)
    az = ch.az
    # ϵ = ch.ϵ
    screen_top = ch.screen_top # λ_top
    screen_bot = ch.screen_bot # λ_bot
    
    α_bot = 1 - az/screen_bot
    α_top = 1 - az/screen_top

    az2 = az^2

    

    @inbounds for ny in 1:Ny, nx in 1:Nxh
        c = 2 + compute_ktilde2(ch, nx, ny)* az2

        diag1 = α_bot - c # Row 1 diagonal
        d = -c #Interior diagonal
        diagN = α_top - c # Row N diagonal

        # Store factors for Thomas algorithm

        # First row

        if nx == 1 && ny == 1
            if screen_bot == Inf && screen_top == Inf
                # (0,0) mode: pin φ̂₁ = 0 to fix the gauge (removes singular
                # null-space of the Neumann Laplacian).  Physical observables
                # depend only on potential *differences*, so the constant is
                # irrelevant while the z-varying part (depolarisation field)
                # is preserved.
                invden[1, 1, 1] = zero(T)
                m_upper[1, 1, 1] = zero(T)
            end
        else
            invden[nx, ny, 1] = inv(diag1)
            m_upper[nx,ny,1] = inv(diag1) # m_upper1 = u1/diag1, u1 = 1 for all modes
        end

        # Interior rows
        for nz in 2:(Nz-1)
            den = d - m_upper[nx,ny,nz-1] #den_n = diag_n - l_n * m_upper_{n-1}
            invden[nx, ny, nz] = inv(den)
            m_upper[nx,ny,nz] = invden[nx, ny, nz] #m_upper_n = u_n / den_n, u_n = 1 for all modes
        end

        # Last row
        invden[nx, ny, Nz] = inv(diagN - m_upper[nx,ny,Nz-1]) # invden_N = 1/(diagN - l_N * m_upper_{N-1}), l_N = 1 for all modes
        m_upper[nx,ny,Nz] = zero(T) # m_upperN = 0 since there is no upper diagonal in the last row
        
    end

    return nothing
end

function precompute_du_self!(c::CoulombHamiltonian)
    T = eltype(c.ρ)
    Nx, Ny, Nz = size(c.ρ)
    Nxh = size(c.ρhat, 1)
    invden = c.inv_den
    m_upper = c.mod_upperd
    scale = -(c.az^2) / (c.ϵ * T(Nx * Ny))

    ρtmp = zeros(T, Nx, Ny, Nz)
    ρhat_tmp = similar(c.ρhat)
    uhat_tmp = similar(c.uhat)
    utmp = zeros(T, Nx, Ny, Nz)
    dptmp = similar(c.dp_scratch)

    fill!(c.du_self, zero(T))

    @inbounds for z in 1:(Nz-1)
        ρtmp[1,1,z] = -one(T)
        ρtmp[1,1,z+1] = one(T)

        mul!(ρhat_tmp, c.Pxy, ρtmp)

        for ny in 1:Ny, nx in 1:Nxh
            if !isfinite(invden[nx, ny, 1]) || !isfinite(invden[nx, ny, Nz])
                uhat_tmp[nx, ny, :] .= zero(Complex{T})
                continue
            end

            dptmp[nx, ny, 1] = (scale * ρhat_tmp[nx, ny, 1]) * invden[nx, ny, 1]
            for nz in 2:Nz
                dptmp[nx, ny, nz] = (scale * ρhat_tmp[nx, ny, nz] - dptmp[nx, ny, nz-1]) * invden[nx, ny, nz]
            end

            uhat_tmp[nx, ny, Nz] = dptmp[nx, ny, Nz]
            for nz in (Nz-1):-1:1
                uhat_tmp[nx, ny, nz] = dptmp[nx, ny, nz] - m_upper[nx, ny, nz] * uhat_tmp[nx, ny, nz+1]
            end
        end

        mul!(utmp, c.iPxy, uhat_tmp)

        c.du_self[z] = utmp[1,1,z+1] - utmp[1,1,z]

        ρtmp[1,1,z] = zero(T)
        ρtmp[1,1,z+1] = zero(T)
    end

    return c.du_self
end

"""
Solve for all modes for a single timestep, reusing precomputed factors.
Then perform inverse FFT to get real-space potential.

Inputs:
  ŝ[kx,ky,n]  (ComplexF64): RHS = -(az^2/ε) ρ̂
Precomputed:
  invden[kx,ky,n] (Float64)
  cp[kx,ky,n]     (Float64)
Output:
  ϕ̂[kx,ky,n] (ComplexF64)
  u[kx,ky,n] (Float64)

This does:
  forward: dp[n] = (ŝ[n] - dp[n-1]) * invden[n]
  backward: ϕ̂[n] = dp[n] - cp[n] * ϕ̂[n+1]
"""

function recalc!(c::CoulombHamiltonian)
    uhat = c.uhat
    rho_hat = c.ρhat
    rho = c.ρ
    Nx, Ny, Nz = size(rho)
    Nxh = size(rho_hat, 1)
    invden = c.inv_den
    m_upper = c.mod_upperd
    dp_scratch = c.dp_scratch
    u = c.u
    T = eltype(u)
    az2 = c.az^2

    # Calculate rho_hat from rho (batched rFFT in x/y for every z plane)
    mul!(rho_hat, c.Pxy, rho)

    @inbounds for ny in 1:Ny, nx in 1:Nxh
        scale = -az2 / (c.ϵ * T(Nx * Ny))
        dp_scratch[nx,ny,1] = (scale * rho_hat[nx,ny,1]) * invden[nx,ny,1]

        # Forward sweep
        for nz in 2:Nz
            dp_scratch[nx,ny,nz] = (scale * rho_hat[nx,ny,nz] - dp_scratch[nx,ny,nz-1]) * invden[nx,ny,nz]
        end

        # Backward substitution
        uhat[nx,ny,Nz] = dp_scratch[nx,ny,Nz]
        for nz in (Nz-1):-1:1
            uhat[nx,ny,nz] = dp_scratch[nx,ny,nz] - m_upper[nx,ny,nz] * uhat[nx,ny,nz+1]
        end
    end

    # Unnormalized inverse rFFT in x/y for every z plane. The spectral RHS is
    # pre-scaled by 1/(Nx*Ny), so `u` is written at the physical scale directly.
    mul!(u, c.iPxy, uhat)

    return u
end

# function ΔH(c::CoulombHamiltonian{T,N}, params, proposal) where {T,N}
@inline function _calculate(::ΔH, c::CoulombHamiltonian, layer::AbstractIsingLayer, proposal)
    T = eltype(c.ρ)
    lattice_size = size(c)
    spin_idx = at_idx(proposal)
    charge_coord_below = idxToCoord(spin_idx, lattice_size)
    charge_coord_above = (charge_coord_below[1], charge_coord_below[2], charge_coord_below[3] + 1)
    scaling = c.scaling[] / c.az
    Δcharge_below = -delta(proposal)*scaling
    Δcharge_above = delta(proposal)*scaling

    # Linear term from the existing potential at the two updated charge planes.
    ΔE_below = Δcharge_below * c.u[charge_coord_below...]
    ΔE_above = Δcharge_above * c.u[charge_coord_above...]

    # Quadratic self term for the local charge-pair update.
    # du_self[z] is precomputed for a unit pair at planes (z, z+1), so this
    # scales with the actual move amplitude as (Δq)^2.
    z = charge_coord_below[3]
    Δq = Δcharge_above
    ΔE_self = (T(0.5) * (Δq^2)) * c.du_self[z]

    return ΔE_below + ΔE_above + ΔE_self
end

@inline function _calculate(::d_iH, c::CoulombHamiltonian, layer::AbstractIsingLayer, proposal::SingleSpinProposal)
    s_idx = @inline at_idx(proposal)
    lattice_size = size(c)
    charge_coord_below = idxToCoord(s_idx, lattice_size)
    charge_coord_above = (charge_coord_below[1], charge_coord_below[2], charge_coord_below[3] + 1)

    # For H = 1/2 * sum_i q_i * ϕ_i and q_pair = (-s, +s) * scaling,
    # the derivative with respect to the dipole variable is Δq/Δs · ϕ.
    return (c.scaling[] / c.az) * (c.u[charge_coord_above...] - c.u[charge_coord_below...])
end

_update!(::Metropolis, c::CoulombHamiltonian, layer::AbstractIsingLayer, proposal::FP) where {FP <: FlipProposal} = begin
    if isaccepted(proposal)
        # 和 ΔH 一样，用自旋所在的 dipole 坐标推两层电荷平面
        spin_idx = at_idx(proposal)
        coord_below = idxToCoord(spin_idx, size(c))
        coord_above = (coord_below[1], coord_below[2], coord_below[3] + 1)

        # dipole → charge 的缩放也要加进来
        scaling = c.scaling[] / c.az
        Δq_below = -delta(proposal) * scaling
        Δq_above =  delta(proposal) * scaling

        # 表面 screening：z=1 和 z=Nz 的电荷需要乘 (1-screening)

        c.ρ[coord_below...] += Δq_below
        c.ρ[coord_above...] += Δq_above
        # 场的重算交给外面的 Recalc 进程周期性处理
    end
    if !isinf(c.recalc_steps) && c.recalc_tracker[] == c.recalc_steps
        recalc!(c)
    end
    if !isinf(c.recalc_steps)
        c.recalc_tracker[] = mod1(c.recalc_tracker[] + 1, Int(c.recalc_steps))
    end
end
