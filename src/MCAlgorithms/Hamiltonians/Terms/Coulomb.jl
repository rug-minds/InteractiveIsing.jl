# TODO: We need a conversion factor from dipole to charge

"""
"""
struct CoulombHamiltonian{T,PT} <: Hamiltonian
    σ::Array{T}                # Charges of the particles
    σhat::Array{Complex{T}}    # Fourier transformed charges
    uhat::Array{Complex{T}}   # Fourier transformed potentials
    # positions::Matrix{T}     # Each column is a position vector
    u::Array{T}                 # Potentials at each site
    ϵ::PT                       # Permittivity constant
end

function CoulombHamiltonian(g::AbstractIsingGraph, eps::T) where {T}
    gdims = size(g[1])
    etype = eltype(g)
    particle_dims = ntuple(i -> i == length(gdims) ? gdims[i] + 1 : gdims[i], length(gdims))
    charges = zeros(etype, particle_dims...)
    ϵ = StaticParam(eps)
    σhat = Array{Complex{etype}}(undef, particle_dims...)
    uhat = Array{Complex{etype}}(undef, particle_dims...)
    CoulombHamiltonian{etype, typeof(ϵ)}(charges,
                             σhat,
                             uhat,
                             zeros(etype, particle_dims...),
                             ϵ)
end


function init!(h::CoulombHamiltonian, g::AbstractIsingGraph)
    h.σ[:,:,:] .= 0
    for dip_idx in size(g, length(size(g))):-1:1
        h.σ[:,:,dip_idx+1] .+= state(g)[:,:,dip_idx]
        h.σ[:,:,dip_idx] .-= state(g)[:,:,dip_idx]
    end
    return h
end


using FFTW
using LinearAlgebra

"""
Compute u[x,y,z] = sum_j h_ij * σ[j]  (Coulomb, PBC x,y, open z)

σ :: Array{Float64,3}  (Nx,Ny,Nz)
returns u :: Array{Float64,3}
"""
function recalc!(c::CoulombHamiltonian)
    σ = c.σ
    u = c.u
    uhat = c.uhat
    σhat = c.σhat

    Nx, Ny, Nz = size(c.σ)

    # --- 1) FFT in x,y for each z ---
    σhat = Array{ComplexF64}(undef, Nx, Ny, Nz)
    for z in 1:Nz
        σhat[:,:,z] .= fft(σ[:,:,z])
    end

    # --- 2) prepare wavevectors ---
    kx(i) = 2π * (i-1 <= Nx÷2 ? i-1 : i-1-Nx) / Nx
    ky(j) = 2π * (j-1 <= Ny÷2 ? j-1 : j-1-Ny) / Ny

    # --- 3) z-coupling for each (kx,ky) ---
    for ix in 1:Nx, iy in 1:Ny
        k = hypot(kx(ix), ky(iy))

        if k == 0.0
            # ---- k = 0 mode ----
            # short-circuit: do nothing (û stays zero)
            # open-circuit depolarization: insert your g0(z-z′) here
            continue
        end

        # precompute kernel along z
        # g = [exp(-k*abs(z-zp)) / (2*k) for z in 1:Nz, zp in 1:Nz]
        g(z,zp) = exp(-k*abs(z-zp)) / (2k)

        # apply z-coupling
        @fastmath @simd for z in 1:Nz
            s = 0.0 + 0.0im
            for zp in 1:Nz
                s += g(z,zp) * σhat[ix,iy,zp]
            end
            uhat[ix,iy,z] = s
        end
    end

    # --- 4) inverse FFT in x,y ---
    @fastmath @simd for z in 1:Nz
        u[:,:,z] .= real(ifft!(@view uhat[:,:,z]))
    end

    return u
end


struct CoulombHamiltonian2{T,PT,P2}
    σ::Array{T,3}                 # optional, can keep if you want
    σhat::Array{Complex{T},3}      # complex charges buffer (real in practice)
    uhat::Array{Complex{T},3}
    u::Array{T,3}
    ϵ::PT

    Pxy::P2                       # plan_fft! for 2D slices
    iPxy::P2                      # plan_ifft! for 2D slices
    kx::Vector{T}
    ky::Vector{T}
end

function CoulombHamiltonian2(g::AbstractIsingGraph, eps::T) where {T}
    gdims = size(g[1])                 # (Nx,Ny,Nz-1)
    etype = eltype(g)

    Nx, Ny, Nz_dip = gdims
    Nz = Nz_dip + 1                    # charge planes

    dims = (Nx, Ny, Nz)

    σ    = zeros(etype, dims...)
    σhat = zeros(Complex{etype}, dims...)
    uhat = zeros(Complex{etype}, dims...)
    u    = zeros(etype, dims...)

    # FFT plans (bind to representative slices)
    sh = @view σhat[:,:,1]
    uh = @view uhat[:,:,1]

    Pxy  = plan_fft!(sh;  flags=FFTW.MEASURE)
    iPxy = plan_ifft!(uh; flags=FFTW.MEASURE)

    # k-vectors (FFT ordering)
    kx = Vector{etype}(undef, Nx)
    ky = Vector{etype}(undef, Ny)

    @inbounds for i in 1:Nx
        ii = i - 1
        kx[i] = 2T(π) * (ii <= Nx ÷ 2 ? ii : ii - Nx) / Nx
    end
    @inbounds for j in 1:Ny
        jj = j - 1
        ky[j] = 2T(π) * (jj <= Ny ÷ 2 ? jj : jj - Ny) / Ny
    end

    ϵ = StaticParam(eps)

    return CoulombHamiltonian{etype,typeof(ϵ),typeof(Pxy)}(
        σ, σhat, uhat, u, ϵ,
        Pxy, iPxy,
        kx, ky
    )
end

function init!(c::CoulombHamiltonian2, g::AbstractIsingGraph)
    σ    = c.σ
    σhat = c.σhat

    Nx, Ny, Nz = size(σ)
    Nz_dip = Nz - 1

    # zero charge buffers
    fill!(σ, zero(eltype(σ)))
    fill!(σhat, zero(eltype(σhat)))

    # accumulate bound charges from dipoles
    @inbounds for z in 1:Nz_dip
        dip = state(g)[:,:,z]    # assumed Array{T,2}
        for j in 1:Ny, i in 1:Nx
            v = dip[i,j]
            σ[i,j,z]   -= v
            σ[i,j,z+1] += v
        end
    end

    # write real charges into complex buffer (imag = 0)
    @inbounds for z in 1:Nz, j in 1:Ny, i in 1:Nx
        σhat[i,j,z] = complex(σ[i,j,z], zero(eltype(σ)))
    end

    return c
end

function recalc!(c::CoulombHamiltonian2{T}) where {T}
    σhat = c.σhat
    uhat = c.uhat
    u    = c.u

    Nx, Ny, Nz = size(σhat)

    # 1) FFT x,y in-place for each z: σhat[:,:,z] := FFT(σhat[:,:,z])
    @inbounds for z in 1:Nz
        sh = @view σhat[:,:,z]
        c.Pxy * sh
    end

    # 2) z-coupling in Fourier domain
    # uhat[ix,iy,z] = Σ_zp exp(-k|z-zp|)/(2k) * σhat[ix,iy,zp]
    @inbounds for iy in 1:Ny
        ky = c.ky[iy]
        for ix in 1:Nx
            kx = c.kx[ix]
            k  = hypot(kx, ky)

            if k == 0
                for z in 1:Nz
                    uhat[ix,iy,z] = 0
                end
                continue
            end

            inv2k = inv(2k)

            # NOTE: still O(Nz^2). Allocation-free, but can be sped up a lot (see below).
            for z in 1:Nz
                s = zero(Complex{T})
                for zp in 1:Nz
                    s += (exp(-k * abs(z - zp)) * inv2k) * σhat[ix,iy,zp]
                end
                uhat[ix,iy,z] = s
            end
        end
    end

    # 3) inverse FFT x,y in-place for each z, copy real part into u
    @inbounds for z in 1:Nz
        uh = @view uhat[:,:,z]
        c.iPxy * uh
        for j in 1:Ny, i in 1:Nx
            u[i,j,z] = real(uh[i,j])
        end
    end

    return u
end