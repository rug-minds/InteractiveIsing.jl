# TODO: We need a conversion factor from dipole to charge
export CoulombHamiltonian, init!, init_tridiag!, recalc!, recalcnew, ΔH, update!

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
    screening::T
    ax::T
    ay::T
    az::T

    Pxy::PxyT                     # plan_rfft for 2D slices
    iPxy::PiT                     # plan_irfft for 2D slices
    kx::Vector{T}
    ky::Vector{T}
    zf::Vector{Complex{T}}        # forward recursion scratch
    zb::Vector{Complex{T}}        # backward recursion scratch
    inv2k::Matrix{T}              # precomputed 1/(2k) for each (kx,ky)
    ez::Matrix{T}                 # precomputed exp(-k*az) for each (kx,ky)
    zf_threads::Matrix{Complex{T}}  # per-thread forward recursion scratch (Nz x nthreads)
    zb_threads::Matrix{Complex{T}}  # per-thread backward recursion scratch (Nz x nthreads)

    tri_cp::Matrix{T}             # precomputed Thomas c' (Nz x (Nxh*Ny))
    tri_invden::Matrix{T}         # precomputed 1/den (Nz x (Nxh*Ny))
    tri_rhsfac::Base.RefValue{T}            # -1/ϵ  (RHS prefactor)
    tri_a_int::Base.RefValue{T}             # interior sub-diagonal coefficient
    tri_c_bc::Base.RefValue{T}              # boundary super-diagonal coefficient
    tri_ϵ::Base.RefValue{T}
    tri_C0::Base.RefValue{T}
    tri_CN::Base.RefValue{T}
    tri_κ::Base.RefValue{T}
    tri_ready::Base.RefValue{Bool}
    tri_k0_neumann::Base.RefValue{Bool}
end

Base.size(c::CoulombHamiltonian) = c.size

function CoulombHamiltonian(
    g::AbstractIsingGraph,
    scaling::Real = 1.f0;
    screening = 0.0
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
    u    = zeros(etype, dims...)
    zf   = Vector{Complex{etype}}(undef, Nz)
    zb   = Vector{Complex{etype}}(undef, Nz)

    # FFT plans (bind to representative slices)
    s  = @view σ[:,:,1]
    uh = @view uhat[:,:,1]

    Pxy  = plan_rfft(s, (1,2);  flags=FFTW.MEASURE)
    iPxy = plan_irfft(uh, Nx, (1,2); flags=FFTW.MEASURE)

    # k-vectors (FFT ordering)
    kx = Vector{etype}(undef, Nxh)
    ky = Vector{etype}(undef, Ny)

    twoπ = etype(2) * etype(π)
    ax = eltype(g)(ax)
    ay = eltype(g)(ay)
    az = eltype(g)(az)
    @inbounds for i in 1:Nxh
        ii = i - 1
        kx[i] = twoπ * ii / (Nx * ax)
    end
    @inbounds for j in 1:Ny
        jj = j - 1
        ky[j] = twoπ * (jj <= Ny ÷ 2 ? jj : jj - Ny) / (Ny * ay)
    end

    inv2k = zeros(etype, Nxh, Ny)
    ez    = zeros(etype, Nxh, Ny)
    @inbounds for iy in 1:Ny
        kyv = ky[iy]
        for ix in 1:Nxh
            if ix == 1 && iy == 1
                continue
            end
            kv = hypot(kx[ix], kyv)
            inv2k[ix, iy] = inv(etype(2) * kv)
            ez[ix, iy] = exp(-kv * az)
        end
    end

    nthreads = Threads.maxthreadid()
    zf_threads = Matrix{Complex{etype}}(undef, Nz, nthreads)
    zb_threads = Matrix{Complex{etype}}(undef, Nz, nthreads)
    nmodes = Nxh * Ny
    tri_cp = Matrix{etype}(undef, Nz, nmodes)
    tri_invden = Matrix{etype}(undef, Nz, nmodes)

    scaling = eltype(g)(scaling)
    scaling = StaticParam(scaling)

    # clamp screening between 0 and 1
    # screening = clamp(screening, 0.0, 1.0)

    c = CoulombHamiltonian{etype,typeof(scaling),typeof(Pxy),typeof(iPxy),length(size(g))}(
        size(g), σ, σhat, uhat, u, scaling, screening,
        ax, ay, az,
        Pxy, iPxy, kx, ky, zf, zb, inv2k, ez, zf_threads, zb_threads,
        tri_cp, tri_invden,
        Ref(zero(etype)), Ref(zero(etype)), Ref(zero(etype)),
        Ref(zero(etype)), Ref(zero(etype)), Ref(zero(etype)), Ref(zero(etype)),
        Ref(false), Ref(false)
    )
    init!(c, g)
    init_tridiag!(c)
    c
end

function init_tridiag!(
    c::CoulombHamiltonian{T};
    ϵ::T = one(T),
    C0::T = c.screening,
    CN::T = c.screening,
    κ::T = zero(T),
) where {T}
    Nx, Ny, Nz = size(c.u)
    Nxh = size(c.σhat, 1)

    inv_az  = inv(c.az)
    inv_az2 = inv_az * inv_az
    α0      = C0 / ϵ
    αN      = CN / ϵ
    κ2      = κ * κ
    tiny    = eps(T)

    a_int = -inv_az2
    c_int = -inv_az2
    c_bc  = -2 * inv_az2
    b_mid = 2 * inv_az2 + κ2
    b_lo  = b_mid + 2 * α0 * inv_az
    b_hi  = b_mid + 2 * αN * inv_az

    mode = 0
    @inbounds for iy in 1:Ny
        ky2 = c.ky[iy] * c.ky[iy]
        for ix in 1:Nxh
            mode += 1
            k2 = c.kx[ix] * c.kx[ix] + ky2

            den = b_lo + k2
            if abs(den) <= tiny
                den += tiny
            end
            invden = inv(den)
            c.tri_invden[1, mode] = invden
            c.tri_cp[1, mode] = c_bc * invden

            for z in 2:(Nz-1)
                den = (b_mid + k2) - a_int * c.tri_cp[z-1, mode]
                if abs(den) <= tiny
                    den += tiny
                end
                invden = inv(den)
                c.tri_invden[z, mode] = invden
                c.tri_cp[z, mode] = c_int * invden
            end

            if Nz > 1
                den = (b_hi + k2) - c_bc * c.tri_cp[Nz-1, mode]
                if abs(den) <= tiny
                    den += tiny
                end
                c.tri_invden[Nz, mode] = inv(den)
            end
        end
    end
    c.tri_rhsfac[] = -inv(ϵ)
    c.tri_a_int[] = a_int
    c.tri_c_bc[] = c_bc
    c.tri_ϵ[] = ϵ
    c.tri_C0[] = C0
    c.tri_CN[] = CN
    c.tri_κ[] = κ
    c.tri_ready[] = true
    c.tri_k0_neumann[] = iszero(κ) && iszero(C0) && iszero(CN)
  
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
    # recalc!(c)
    recalc!(c)
    return c
end

function recalc!(c::CoulombHamiltonian{T}) where {T}
    σ    = c.σ
    σhat = c.σhat
    uhat = c.uhat
    u    = c.u
    zf   = c.zf
    zb   = c.zb

    Nx, Ny, Nz = size(u)
    Nxh        = size(σhat, 1)

    # 1) rFFT x,y for each z: σhat[:,:,z] := rfft(σ[:,:,z])
    @inbounds for z in 1:Nz
        s  = @view σ[:, :, z]
        sh = @view σhat[:, :, z]
        mul!(sh, c.Pxy, s)
    end

    # 2) z-coupling in Fourier domain for k>0 modes
    #    uhat[ix,iy,z] = Σ_zp exp(-k|z-zp|)/(2k) * σhat[ix,iy,zp]
    @inbounds begin
        fill!((@view uhat[1,1,:]), zero(Complex{T}))   # k=0 模式稍后单独处理
        for iy in 1:Ny
            ky = c.ky[iy]
            for ix in 1:Nxh
                if ix == 1 && iy == 1
                    continue                         # (kx,ky) = (0,0) 留给专门的 k=0 处理
                end
                kx = c.kx[ix]
                k  = hypot(kx, ky)

                inv2k = inv(2k)
                e     = exp(-k * c.az)

                zf[1] = σhat[ix,iy,1]
                @inbounds for z in 2:Nz
                    zf[z] = σhat[ix,iy,z] + e * zf[z-1]
                end
                zb[Nz] = σhat[ix,iy,Nz]
                @inbounds for z in (Nz-1):-1:1
                    zb[z] = σhat[ix,iy,z] + e * zb[z+1]
                end

                @inbounds for z in 1:Nz
                    # Use convention E = -1/2 Σ σ_i u_i, so u = -Kσ
                    uhat[ix,iy,z] = -(zf[z] + zb[z] - σhat[ix,iy,z]) * inv2k
                end
            end
        end
    end

    # 3) k=0 模式的 z 方向卷积：G0(z,z') ~ -|z-z'|/2（差一个常数对 ΔH 无影响）
    @inbounds begin
        sh0 = @view σhat[1,1,:]    # (kx,ky) = (0,0) 的面电荷谱
        uh0 = @view uhat[1,1,:]

        for z in 1:Nz
            acc = zero(Complex{T})
            for zp in 1:Nz
                # 物理距离 |z-z'| * az，对 kernel 取 -|z-z'|/2
                dist = T(abs(z - zp)) * c.az
                acc += (dist * T(0.5)) * sh0[zp]
            end
            uh0[z] = acc
        end
    end

    # 4) inverse rFFT x,y for each z, write directly into u
    @inbounds for z in 1:Nz
        uh = @view uhat[:, :, z]
        uz = @view u[:, :, z]
        mul!(uz, c.iPxy, uh)
    end

    return u
end

function tridiag_recalc!(c::CoulombHamiltonian{T},   
    ϵ::T = one(T),
    C0::T = c.screening,
    CN::T = c.screening,
    λ::T = zero(T)) where {T}

    return nothing
end

# function recalc_tridiag!(
#     c::CoulombHamiltonian{T};
#     ϵ::T = one(T),
#     C0::T = c.screening,
#     CN::T = c.screening,
#     κ::T = zero(T),
# ) where {T}
#     σ    = c.σ
#     σhat = c.σhat
#     uhat = c.uhat
#     u    = c.u

#     _, Ny, Nz = size(u)
#     Nxh        = size(σhat, 1)

#     # 1) rFFT in (x,y) for each z-plane.
#     @inbounds for z in 1:Nz
#         s  = @view σ[:, :, z]
#         sh = @view σhat[:, :, z]
#         mul!(sh, c.Pxy, s)
#     end

#     if !c.tri_ready[] || c.tri_ϵ[] != ϵ || c.tri_C0[] != C0 || c.tri_CN[] != CN || c.tri_κ[] != κ
#         init_tridiag!(c; ϵ = ϵ, C0 = C0, CN = CN, κ = κ)
#     end

#     dp = c.zb
#     rhsfac = c.tri_rhsfac[]
#     a_int = c.tri_a_int[]
#     c_bc = c.tri_c_bc[]
#     tri_cp = c.tri_cp
#     tri_invden = c.tri_invden
#     k0_neumann = c.tri_k0_neumann[]

#     # For pure Neumann BCs at k=0, enforce compatibility and fix gauge.
#     if k0_neumann
#         sh0 = @view σhat[1, 1, :]
#         qsum = zero(Complex{T})
#         @inbounds for z in 1:Nz
#             qsum += sh0[z]
#         end
#         if qsum != zero(Complex{T})
#             qcorr = qsum / T(Nz)
#             @inbounds for z in 1:Nz
#                 sh0[z] -= qcorr
#             end
#         end
#     end

#     mode = 0
#     if Nz == 1
#         @inbounds for iy in 1:Ny
#             for ix in 1:Nxh
#                 mode += 1
#                 uhat[ix, iy, 1] = (rhsfac * σhat[ix, iy, 1]) * tri_invden[1, mode]
#             end
#         end
#     else
#         @inbounds for iy in 1:Ny
#             for ix in 1:Nxh
#                 mode += 1

#                 dp[1] = (rhsfac * σhat[ix, iy, 1]) * tri_invden[1, mode]

#                 # interior rows
#                 for z in 2:(Nz-1)
#                     dp[z] = (rhsfac * σhat[ix, iy, z] - a_int * dp[z-1]) * tri_invden[z, mode]
#                 end

#                 dp[Nz] = (rhsfac * σhat[ix, iy, Nz] - c_bc * dp[Nz-1]) * tri_invden[Nz, mode]

#                 # backward substitution
#                 uhat[ix, iy, Nz] = dp[Nz]
#                 for z in (Nz-1):-1:1
#                     uhat[ix, iy, z] = dp[z] - tri_cp[z, mode] * uhat[ix, iy, z+1]
#                 end
#             end
#         end
#     end

#     if k0_neumann
#         uh0 = @view uhat[1, 1, :]
#         u0mean = zero(Complex{T})
#         @inbounds for z in 1:Nz
#             u0mean += uh0[z]
#         end
#         u0mean /= T(Nz)
#         @inbounds for z in 1:Nz
#             uh0[z] -= u0mean
#         end
#     end

#     # 3) inverse rFFT in (x,y) for each z-plane.
#     @inbounds for z in 1:Nz
#         uh = @view uhat[:, :, z]
#         uz = @view u[:, :, z]
#         mul!(uz, c.iPxy, uh)
#     end

#     return u
# end

function ΔH(c::CoulombHamiltonian{T,N}, params, proposal) where {T,N}
    # println("Calculating ΔH for CoulombHamiltonian ")
    lattice_size = size(c)
    spin_idx = at_idx(proposal)
    charge_coord_below = idxToCoord(spin_idx, lattice_size)
    charge_coord_above = (charge_coord_below[1], charge_coord_below[2], charge_coord_below[3] + 1)
    Δcharge_below = -delta(proposal)*c.scaling[]
    Δcharge_above = delta(proposal)*c.scaling[]

    ΔE_below = -Δcharge_below * c.u[charge_coord_below...]
    ΔE_above = -Δcharge_above * c.u[charge_coord_above...]
    return ΔE_below + ΔE_above
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
