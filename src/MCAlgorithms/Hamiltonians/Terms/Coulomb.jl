# TODO: We need a conversion factor from dipole to charge
export CoulombHamiltonian, init!, precompute_du_self!, recalc!, recalcnew, ΔH, update!

function mygemmavx!(C, A, B)
    @turbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end
struct CoulombHamiltonian{T,PT,PxyT,PiT,N} <: HamiltonianTerm
    size::NTuple{N,Int}
    σ::Array{T,3}                 # real charges
    σhat::Array{Complex{T},3}     # rfft(σ) over (x,y)
    uhat::Array{Complex{T},3}     # spectral potential (same size as σhat)
    u::Array{T,3}                 # real potential
    scaling::PT
    screen_top::T
    screen_bot::T
    ax::T
    ay::T
    az::T
    ϵ::T
    du_self::Array{T,1}           # per-layer unit charge-pair field jump du[z+1]-du[z]

    Pxy::PxyT                     # plan_rfft for 2D slices
    iPxy::PiT                     # plan_irfft for 2D slices
    
    mod_upperd::Array{T, 3}       # Thomas algorithm modified upper diagonal (cp), size (Nxh, Ny, Nz)
    inv_den::Array{T, 3}
    dp_scratch::Array{Complex{T}, 3}       # Thomas algorithm forward sweep scratch space
end

Base.size(c::CoulombHamiltonian) = c.size

function CoulombHamiltonian(
    g::AbstractIsingGraph,
    scaling::Real = 1.f0;
    screening = Inf32,
    screen_len_top = screening,
    screen_len_bot = screening
)
    gdims = size(g[1])                 # (Nx,Ny,Nz-1)
    etype = eltype(g)

    ax, ay, az = lattice_constants(top(g[1]))

    Nx, Ny, Nz_dip = gdims
    Nz = Nz_dip + 1                    # charge planes

    dims = (Nx, Ny, Nz)

    σ    = zeros(etype, dims...)
    Nxh  = Nx ÷ 2 + 1
    σhat = zeros(Complex{etype}, Nxh, Ny, Nz)
    uhat = zeros(Complex{etype}, Nxh, Ny, Nz)


    # FFT plans (bind to representative slices)
    s  = @view σ[:,:,1]
    uh = @view uhat[:,:,1]

    Pxy  = plan_rfft(s, (1,2);  flags=FFTW.MEASURE)
    iPxy = plan_irfft(uh, Nx, (1,2); flags=FFTW.MEASURE)

    twoπ = etype(2) * etype(π)
    ax = eltype(g)(ax)
    ay = eltype(g)(ay)
    az = eltype(g)(az)

    # Precompute Thomas algorithm factors for each (kx,ky) mode
    mod_upperd = zeros(etype, Nxh, Ny, Nz) # modified upper diagonal
    inv_den = zeros(etype, Nxh, Ny, Nz)    # inverse of main diagonal after forward elimination
    dp_scratch = zeros(Complex{etype}, Nxh, Ny, Nz) # scratch space for forward sweep
    du_self = zeros(etype, Nz_dip)

    scaling = eltype(g)(scaling)
    scaling = StaticParam(scaling)

    c = CoulombHamiltonian{etype, typeof(scaling), typeof(Pxy), typeof(iPxy), 3}(
        dims, σ, σhat, uhat, zeros(etype, dims...), scaling, screen_len_top, screen_len_bot, ax, ay, az, 1f0, du_self,
        Pxy, iPxy,
        mod_upperd,
        inv_den,
        dp_scratch
    )
    init!(c, g)
    return c
   
end

function init!(c::CoulombHamiltonian, g::AbstractIsingGraph)
    σ    = c.σ

    Nx, Ny, Nz = size(σ)
    Nz_dip = Nz - 1

    # zero charge buffers
    fill!(σ, zero(eltype(σ)))

    # accumulate bound charges from dipoles
    @inbounds for z in 1:Nz_dip
        
        dip = state(g)[:,:,z]    # assumed Array{T,2}
        @inbounds for j in 1:Ny, i in 1:Nx
            v = dip[i,j]
            v = v * c.scaling[] # Scaling factor dipole to charge
            σ[i,j,z]   -= v
            σ[i,j,z+1] += v
        end
    end
    precompute_solve_factors!(c)
    precompute_du_self!(c)
    recalc!(c)

    return c
end


function compute_ktilde2(c::CoulombHamiltonian{T}, nx, ny) where {T}
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
function precompute_solve_factors!(ch::CoulombHamiltonian{T}) where T
    m_upper = ch.mod_upperd #modified upper diagonal,
    invden = ch.inv_den
    Nx, _, _ = size(ch.σ)
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
        invden[nx, ny, 1] = inv(diag1)
        m_upper[nx,ny,1] = inv(diag1) # m_upper1 = u1/diag1, u1 = 1 for all modes

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

function precompute_du_self!(c::CoulombHamiltonian{T}) where {T}
    Nx, Ny, Nz = size(c.σ)
    Nxh = size(c.σhat, 1)
    invden = c.inv_den
    m_upper = c.mod_upperd
    scale = -(c.az^2) / c.ϵ

    σtmp = zeros(T, Nx, Ny, Nz)
    σhat_tmp = similar(c.σhat)
    uhat_tmp = similar(c.uhat)
    utmp = zeros(T, Nx, Ny, Nz)
    dptmp = similar(c.dp_scratch)

    fill!(c.du_self, zero(T))

    @inbounds for z in 1:(Nz-1)
        σtmp[1,1,z] = -one(T)
        σtmp[1,1,z+1] = one(T)

        for zz in 1:Nz
            s = @view σtmp[:, :, zz]
            sh = @view σhat_tmp[:, :, zz]
            mul!(sh, c.Pxy, s)
        end

        for ny in 1:Ny, nx in 1:Nxh
            if !isfinite(invden[nx, ny, 1]) || !isfinite(invden[nx, ny, Nz])
                uhat_tmp[nx, ny, :] .= zero(Complex{T})
                continue
            end

            dptmp[nx, ny, 1] = (scale * σhat_tmp[nx, ny, 1]) * invden[nx, ny, 1]
            for nz in 2:Nz
                dptmp[nx, ny, nz] = (scale * σhat_tmp[nx, ny, nz] - dptmp[nx, ny, nz-1]) * invden[nx, ny, nz]
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

        σtmp[1,1,z] = zero(T)
        σtmp[1,1,z+1] = zero(T)
    end

    return c.du_self
end

"""
Solve for all modes for a single timestep, reusing precomputed factors.
Then perform inverse FFT to get real-space potential.

Inputs:
  ŝ[kx,ky,n]  (ComplexF64): RHS = -(az^2/ε) σ̂
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

function recalc!(c::CoulombHamiltonian{T}) where {T}
    uhat = c.uhat
    sigma_hat = c.σhat
    sigma = c.σ
    Nx, Ny, Nz = size(uhat)
    Nxh = size(sigma_hat, 1)
    invden = c.inv_den
    m_upper = c.mod_upperd
    dp_scratch = c.dp_scratch
    u = c.u
    az2 = c.az^2

    # Calculate sigma_hat from sigma (rFFT in x,y)
    @inbounds for z in 1:Nz
        s  = @view sigma[:, :, z]
        sh = @view sigma_hat[:, :, z]
        mul!(sh, c.Pxy, s)
    end

    @inbounds for ny in 1:Ny, nx in 1:Nxh
        scale = -az2 / c.ϵ
        dp_scratch[nx,ny,1] = (scale * sigma_hat[nx,ny,1]) * invden[nx,ny,1]

        # Forward sweep
        for nz in 2:Nz
            dp_scratch[nx,ny,nz] = (scale * sigma_hat[nx,ny,nz] - dp_scratch[nx,ny,nz-1]) * invden[nx,ny,nz]
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

function ΔH(c::CoulombHamiltonian{T,N}, params, proposal) where {T,N}
    # println("Calculating ΔH for CoulombHamiltonian ")
    lattice_size = size(c)
    spin_idx = at_idx(proposal)
    charge_coord_below = idxToCoord(spin_idx, lattice_size)
    charge_coord_above = (charge_coord_below[1], charge_coord_below[2], charge_coord_below[3] + 1)
    Δcharge_below = -delta(proposal)*c.scaling[]
    Δcharge_above = delta(proposal)*c.scaling[]

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

update!(::Metropolis, c::CoulombHamiltonian{T}, context) where {T} = begin
    (;proposal) = context
    if isaccepted(proposal)
        # 和 ΔH 一样，用自旋所在的 dipole 坐标推两层电荷平面
        spin_idx = at_idx(proposal)
        coord_below = idxToCoord(spin_idx, size(c))
        coord_above = (coord_below[1], coord_below[2], coord_below[3] + 1)

        # dipole → charge 的缩放也要加进来
        Δq_below = -delta(proposal) * c.scaling[]
        Δq_above =  delta(proposal) * c.scaling[]

        # 表面 screening：z=1 和 z=Nz 的电荷需要乘 (1-screening)
        Nz = size(c.σ, 3)   # 真实电荷平面的层数（比 dipole 多 1 层）
        # if coord_below[3] == 1
        #     Δq_below *= (1 - c.screening)
        # end
        # if coord_above[3] == Nz
        #     Δq_above *= (1 - c.screening)
        # end

        c.σ[coord_below...] += Δq_below
        c.σ[coord_above...] += Δq_above
        # 场的重算交给外面的 Recalc 进程周期性处理
    end
end
