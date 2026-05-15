# TODO: We need a conversion factor from dipole to charge
export CoulombHamiltonian, init!, precompute_du_self!, recalc!, recalcnew, ΔH, update!

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
struct CoulombInternal{T,PxyT,IPxyT,N} <: InternalImplementation
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

    Pxy::PxyT                     # plan_rfft for 2D slices
    iPxy::IPxyT                   # plan_irfft for 2D slices

    mod_upperd::Array{T,3}        # Thomas algorithm modified upper diagonal (cp), size (Nxh, Ny, Nz)
    inv_den::Array{T,3}
    dp_scratch::Array{Complex{T},3} # Thomas algorithm forward sweep scratch space

    recalc_steps::Int
    recalc_tracker::Base.RefValue{Int} # Counter to track when to recalculate potential (for external coupling)
end

struct CoulombHamiltonian{P,I} <: LayerTerm
    layer::Int
    parameters::P
    internal::I
end

Base.size(c::CoulombHamiltonian) = c.size

@inline function CoulombHamiltonian(;
    layer = 1,
    scaling = 1.f0,
    screening = Inf32,
    screen_len_top = screening,
    screen_len_bot = screening,
    recalc = 1
)
    params = Parameters(
        parameter(;
            scaling,
            type = AbstractArray,
            default = ConstVal(1f0),
            ensure = ensure_isinggraph_scalar,
            info = "Dipole-to-charge scaling factor",
        ),
    )
    internal = InternalPlan((; screen_len_top, screen_len_bot, recalc)) do plan, g
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

        return CoulombInternal(
            charge_size,                                  # size
            ρ,                                            # ρ
            ρhat,                                         # ρhat
            uhat,                                         # uhat
            u,                                            # u
            T(config.screen_len_top),                     # screen_top
            T(config.screen_len_bot),                     # screen_bot
            T(constants[1]),                              # ax
            T(constants[2]),                              # ay
            T(constants[3]),                              # az
            one(T),                                       # ϵ
            zeros(T, Nz_dip),                             # du_self
            plan_rfft(@view(ρ[:, :, 1]), (1, 2); flags = FFTW.MEASURE), # Pxy
            plan_irfft(@view(uhat[:, :, 1]), Nx, (1, 2); flags = FFTW.MEASURE), # iPxy
            zeros(T, spectral_size),                      # mod_upperd
            zeros(T, spectral_size),                      # inv_den
            zeros(Complex{T}, spectral_size),             # dp_scratch
            config.recalc,                                # recalc_steps
            Ref{Int}(1),                                  # recalc_tracker
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
    recalc = 1
)
    h = instantiate(
        CoulombHamiltonian(;
            layer,
            scaling,
            screening,
            screen_len_top,
            screen_len_bot,
            recalc,
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
    ρ    = c.ρ
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
    precompute_solve_factors!(c)
    precompute_du_self!(c)
    recalc!(c)

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
    scale = -(c.az^2) / c.ϵ

    ρtmp = zeros(T, Nx, Ny, Nz)
    ρhat_tmp = similar(c.ρhat)
    uhat_tmp = similar(c.uhat)
    utmp = zeros(T, Nx, Ny, Nz)
    dptmp = similar(c.dp_scratch)

    fill!(c.du_self, zero(T))

    @inbounds for z in 1:(Nz-1)
        ρtmp[1,1,z] = -one(T)
        ρtmp[1,1,z+1] = one(T)

        for zz in 1:Nz
            s = @view ρtmp[:, :, zz]
            sh = @view ρhat_tmp[:, :, zz]
            mul!(sh, c.Pxy, s)
        end

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

        for zz in 1:Nz
            uh = @view uhat_tmp[:, :, zz]
            uview = @view utmp[:, :, zz]
            mul!(uview, c.iPxy, uh)
        end

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
    Nx, Ny, Nz = size(uhat)
    Nxh = size(rho_hat, 1)
    invden = c.inv_den
    m_upper = c.mod_upperd
    dp_scratch = c.dp_scratch
    u = c.u
    T = eltype(u)
    az2 = c.az^2

    # Calculate rho_hat from rho (rFFT in x,y)
    @inbounds for z in 1:Nz
        s  = @view rho[:, :, z]
        sh = @view rho_hat[:, :, z]
        mul!(sh, c.Pxy, s)
    end

    @inbounds for ny in 1:Ny, nx in 1:Nxh
        scale = -az2 / c.ϵ
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

    # Inverse rFFT in x,y for each z-plane
    @inbounds for z in 1:Nz
        uh = @view uhat[:, :, z]
        uview = @view u[:, :, z]
        mul!(uview, c.iPxy, uh)
    end

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

@inline function _calculate(::d_iH, c::CoulombHamiltonian, layer::AbstractIsingLayer, s_idx)
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
    if c.recalc_tracker[] == c.recalc_steps
        recalc!(c)
    end
    c.recalc_tracker[] = mod1(c.recalc_tracker[] + 1, c.recalc_steps)
end
