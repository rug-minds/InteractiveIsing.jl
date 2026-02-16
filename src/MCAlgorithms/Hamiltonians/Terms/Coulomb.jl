# TODO: We need a conversion factor from dipole to charge
export CoulombHamiltonian, init!, recalc!, recalcnew, ΔH, update!

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
    zf_threads::Vector{Vector{Complex{T}}}  # per-thread forward recursion scratch
    zb_threads::Vector{Vector{Complex{T}}}  # per-thread backward recursion scratch
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

    nthreads = Threads.nthreads()
    zf_threads = [Vector{Complex{etype}}(undef, Nz) for _ in 1:nthreads]
    zb_threads = [Vector{Complex{etype}}(undef, Nz) for _ in 1:nthreads]

    scaling = eltype(g)(scaling)
    scaling = StaticParam(scaling)

    # clamp screening between 0 and 1
    screening = clamp(screening, 0.0, 1.0)

    c = CoulombHamiltonian{etype,typeof(scaling),typeof(Pxy),typeof(iPxy),length(size(g))}(
        size(g), σ, σhat, uhat, u, scaling, screening,
        ax, ay, az,
        Pxy, iPxy, kx, ky, zf, zb, inv2k, ez, zf_threads, zb_threads
    )
    init!(c, g)
    c
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
            # if z == 1
            #     vscreened = v * (1 - c.screening)
            #     σ[i,j,z]   -= vscreened
            #     σ[i,j,z+1] += v
            # elseif z == Nz_dip
            #     vscreened = v * (1 - c.screening)
            #     σ[i,j,z]   -= v
            #     σ[i,j,z+1] += vscreened
            # else
            #     σ[i,j,z]   -= v
            #     σ[i,j,z+1] += v
            # end
        end
    end
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
