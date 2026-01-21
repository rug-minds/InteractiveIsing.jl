using FFTW
using LinearAlgebra
using GLMakie

"""
Compute depolarization potential phi(x,y,z) and Ez(x,y,z) for a thin film
using 2D FFT in (x,y) and an analytic k-space kernel along z.

Pz: (Nx,Ny,Nz) polarization field (e.g. Pz = p0 .* s)
Returns: phi, Ez, sigmas, z_sheets
"""
function depolar_from_Pz(Pz;
    eps::Float64 = 1.0,
    dz::Float64  = 1.0,
    include_interfaces::Bool = true,
    include_surfaces::Bool   = true,
    k0_mode_zero::Bool       = true
)
    Nx, Ny, Nz = size(Pz)

    # ----- k grid (radians / lattice unit). assume dx=dy=1 -----
    kx = 2π .* FFTW.fftfreq(Nx, 1.0)
    ky = 2π .* FFTW.fftfreq(Ny, 1.0)

    lambda = 5.0        # screening length
    kappa  = 1.0 / lambda

    Keff = Array{Float64}(undef, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        Keff[i,j] = sqrt(Keff[i,j]^2 + kappa^2)
    end


    # ----- build charge sheets sigma(x,y; sheet) and their z positions -----
    # We'll store sheets in 3rd dimension
    # Count number of sheets
    Nsheets = 0
    if include_surfaces
        Nsheets += 2
    end
    if include_interfaces && Nz >= 2
        Nsheets += (Nz - 1)
    end

    sigmas = Array{Float64}(undef, Nx, Ny, Nsheets)
    z_sheets = Array{Float64}(undef, Nsheets)

    sheet = 1

    # bottom surface at z=0: sigma = -Pz[:,:,1]
    if include_surfaces
        sigmas[:,:,sheet] .= -Pz[:,:,1]
        z_sheets[sheet] = 0.0
        sheet += 1
    end

    # internal interfaces at z=(i-0.5)*dz for i=2..Nz (between i-1 and i)
    # equivalently i=1..Nz-1 : z=(i-0.5)*dz with P(i+1)-P(i)
    if include_interfaces && Nz >= 2
        for i in 1:(Nz-1)
            sigmas[:,:,sheet] .= -(Pz[:,:,i+1] .- Pz[:,:,i])
            z_sheets[sheet] = (i - 0.5) * dz
            sheet += 1
        end
    end

    # top surface at z=(Nz-1)*dz: sigma = +Pz[:,:,Nz]
    if include_surfaces
        sigmas[:,:,sheet] .= +Pz[:,:,Nz]
        z_sheets[sheet] = (Nz - 1) * dz
        sheet += 1
    end

    # ----- FFT of each sheet in (x,y) -----
    # Use complex arrays
    sigma_k = Array{ComplexF64}(undef, Nx, Ny, Nsheets)
    for s in 1:Nsheets
        sigma_k[:,:,s] = fft(sigmas[:,:,s])  # 2D FFT
    end

    # ----- prefactor 1/(2 eps |k|), set k=0 mode to 0 -----
    pref = Array{Float64}(undef, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        if Keff[i,j] == 0.0
            pref[i,j] = 0.0
        else
            pref[i,j] = 1.0 / (2.0 * eps * Keff[i,j])
        end
    end

    # ----- compute phi_k for each layer -----
    z_layers = (0:(Nz-1)) .* dz
    phi_k = zeros(ComplexF64, Nx, Ny, Nz)

    for iz in 1:Nz
        z = z_layers[iz]
        # accumulate over sheets
        @inbounds for s in 1:Nsheets
            dist = abs(z - z_sheets[s])
            # W = exp(-|k|*dist) as an (Nx,Ny) factor
            # do elementwise: phi_k[:,:,iz] += pref .* exp.(-K*dist) .* sigma_k[:,:,s]
            # We'll do it with loops for speed and to avoid allocations
            for i in 1:Nx, j in 1:Ny
                w = exp(-Keff[i,j] * dist)
                phi_k[i,j,iz] += (pref[i,j] * w) * sigma_k[i,j,s]
            end
        end

        if k0_mode_zero
            phi_k[1,1,iz] = 0.0 + 0.0im  # kx=0,ky=0 mode in FFT indexing
        end
    end

    # ----- inverse FFT back to real space -----
    phi = Array{Float64}(undef, Nx, Ny, Nz)
    for iz in 1:Nz
        phi[:,:,iz] = real(ifft(phi_k[:,:,iz]))
    end

    # ----- Ez by finite difference along z -----
    Ez = zeros(Float64, Nx, Ny, Nz)
    if Nz >= 2
        # one-sided at boundaries
        Ez[:,:,1]  .= -(phi[:,:,2] .- phi[:,:,1]) ./ dz
        Ez[:,:,Nz] .= -(phi[:,:,Nz] .- phi[:,:,Nz-1]) ./ dz
        # central inside
        for iz in 2:(Nz-1)
            Ez[:,:,iz] .= -(phi[:,:,iz+1] .- phi[:,:,iz-1]) ./ (2dz)
        end
    end

    return phi, Ez, sigmas, z_sheets
end


# ----------------------------
# Quick demo (like your size)
# ----------------------------

Nx, Ny, Nz = 40, 40, 8
# random Ising polarization
# s = rand([-1.0, 1.0], Nx, Ny, Nz)
s = ones(Float64, Nx, Ny, Nz)
s[:, :, 1] .= rand(0:1, Nx, Ny)
s[:, :, Nz] .= rand(0:1, Nx, Ny)

Pz = s  # p0=1

phi, Ez, sigmas, z_sheets = depolar_from_Pz(Pz; eps=1.0, dz=1.0,
                                            include_interfaces=true,
                                            include_surfaces=true,
                                            k0_mode_zero=true)

println("phi size = ", size(phi), ", Ez size = ", size(Ez))
println("Nsheets = ", size(sigmas,3), ", first z_sheets = ", z_sheets[1:min(end,5)])

Nx, Ny, Nz = size(Ez)


fig = Figure(size=(800,600))
ax = Axis3(fig[1,1],
    xlabel="x", ylabel="y", zlabel="z",
    title="3D Ez volume"
)

vol = volume!(
    ax,
    Ez,                  # 3D array
    algorithm = :mip,    # maximum intensity projection
    colormap = :balance
)

Colorbar(fig[1,2], vol, label="Ez")
display(fig)


# Uncomment to run demo:



