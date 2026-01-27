using FFTW
using LinearAlgebra
using Random

"""
Monte Carlo simulation with exact depolarization energy calculation using 
precompiled k-space kernel method.

This approach avoids the frozen-field approximation by:
1. Precomputing the electrostatic kernel K(kx,ky) in k-space
2. Tracking charge sheets σ̃(k) in Fourier space
3. Computing exact ΔH for each flip using quadratic form without FFT

Key advantage: Correctly handles field reorganization for surface flips.
"""

# ============================================================================
# Structure to hold precomputed kernel and current state
# ============================================================================
mutable struct DepolMCState
    # System size
    Nx::Int
    Ny::Int
    Nz::Int
    
    # Spin configuration (±1)
    spins::Array{Float64, 3}  # (Nx, Ny, Nz)
    
    # k-space grids
    kx::Vector{Float64}  # kx frequencies
    ky::Vector{Float64}  # ky frequencies
    
    # Precomputed kernel K(kx, ky, s, s') for all k-points and sheet pairs
    # Dimensions: (Nx, Ny, Nsheets, Nsheets)
    K_kernel::Array{ComplexF64, 4}
    
    # Current charge sheet amplitudes in k-space
    # Dimensions: (Nx, Ny, Nsheets)
    sigma_k::Array{ComplexF64, 3}
    
    # Sheet positions in z
    z_sheets::Vector{Float64}
    
    # Which sheets are affected by flips at each z-layer
    # flip_map[iz] = list of (sheet_index, Delta_sigma) pairs
    flip_map::Vector{Vector{Tuple{Int, Float64}}}
    
    # Physical parameters
    eps::Float64      # dielectric constant
    dz::Float64       # z-spacing
    kappa::Float64    # screening wavevector (1/lambda)
    
    # Current total depolarization energy
    H_depol::Float64
end


# ============================================================================
# Initialize the Monte Carlo state with precomputed kernel
# ============================================================================
"""
    initialize_depol_mc(Nx, Ny, Nz; eps=1.0, dz=1.0, lambda=Inf, 
                        include_interfaces=true, include_surfaces=true)

Precompute the k-space kernel and initialize tracking structures.

Returns: DepolMCState with precomputed kernel and initial configuration.
"""
function initialize_depol_mc(Nx::Int, Ny::Int, Nz::Int;
    eps::Float64 = 1.0,
    dz::Float64  = 1.0,
    lambda::Float64 = Inf,  # screening length (Inf = no screening)
    include_interfaces::Bool = true,
    include_surfaces::Bool   = true,
    initial_spins = nothing  # Can provide initial spin config
)
    # ----- k-space grid -----
    kx = 2π .* FFTW.fftfreq(Nx, 1.0)
    ky = 2π .* FFTW.fftfreq(Ny, 1.0)
    
    kappa = 1.0 / lambda
    
    # Effective wavevector including screening
    Keff = Array{Float64}(undef, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        k2 = kx[i]^2 + ky[j]^2
        Keff[i,j] = sqrt(k2 + kappa^2)
    end
    
    # ----- Build list of charge sheets -----
    Nsheets = 0
    if include_surfaces
        Nsheets += 2
    end
    if include_interfaces && Nz >= 2
        Nsheets += (Nz - 1)
    end
    
    z_sheets = zeros(Float64, Nsheets)
    sheet = 1
    
    # Bottom surface at z=0
    if include_surfaces
        z_sheets[sheet] = 0.0
        sheet += 1
    end
    
    # Internal interfaces at z=(i-0.5)*dz, between layers i and i+1
    if include_interfaces && Nz >= 2
        for i in 1:(Nz-1)
            z_sheets[sheet] = (i - 0.5) * dz
            sheet += 1
        end
    end
    
    # Top surface at z=(Nz-1)*dz
    if include_surfaces
        z_sheets[sheet] = (Nz - 1) * dz
        sheet += 1
    end
    
    # ----- Precompute kernel K(kx, ky, s, s') -----
    # This is the electrostatic Green's function in k-space
    # K_ss' = (1 / 2ε|k|) * exp(-|k| * |z_s - z_s'|)
    
    println("Precomputing k-space kernel...")
    K_kernel = zeros(ComplexF64, Nx, Ny, Nsheets, Nsheets)
    
    @inbounds for i in 1:Nx, j in 1:Ny
        keff = Keff[i,j]
        
        # Regularize k=0 mode
        if keff == 0.0
            # k=0 mode contributes nothing (gauge freedom)
            continue
        end
        
        prefactor = 1.0 / (2.0 * eps * keff)
        
        for s in 1:Nsheets, s2 in 1:Nsheets
            dist = abs(z_sheets[s] - z_sheets[s2])
            K_kernel[i,j,s,s2] = prefactor * exp(-keff * dist)
        end
    end
    
    println("Kernel precomputed. Size: $(size(K_kernel))")
    
    # ----- Build flip map: which sheets are affected by flip at layer iz -----
    # This depends on geometry:
    # - Flip at iz=1 (bottom): affects bottom surface + first interface
    # - Flip at iz=Nz (top): affects top surface + last interface  
    # - Flip at interior iz: affects two adjacent interfaces
    
    flip_map = Vector{Vector{Tuple{Int, Float64}}}(undef, Nz)
    
    for iz in 1:Nz
        affected = Tuple{Int, Float64}[]
        
        if include_surfaces && iz == 1
            # Bottom surface sheet
            sheet_idx = 1
            push!(affected, (sheet_idx, -1.0))  # Δσ = -ΔPz
        end
        
        if include_interfaces && iz >= 1 && iz < Nz
            # Interface between iz and iz+1
            # Position: z=(iz-0.5)*dz
            # This is interface number iz
            sheet_idx = include_surfaces ? iz + 1 : iz
            push!(affected, (sheet_idx, -1.0))  # Δσ = -(Pz[iz+1] - Pz[iz])
        end
        
        if include_interfaces && iz >= 2
            # Interface between iz-1 and iz
            # This is interface number iz-1
            sheet_idx = include_surfaces ? iz : iz - 1
            push!(affected, (sheet_idx, +1.0))  # Δσ = +(Pz[iz] - Pz[iz-1])
        end
        
        if include_surfaces && iz == Nz
            # Top surface sheet
            sheet_idx = Nsheets
            push!(affected, (sheet_idx, +1.0))  # Δσ = +ΔPz
        end
        
        flip_map[iz] = affected
    end
    
    # ----- Initialize spin configuration -----
    if initial_spins === nothing
        spins = ones(Float64, Nx, Ny, Nz)
    else
        spins = copy(initial_spins)
    end
    
    # ----- Compute initial charge sheets and FFT -----
    sigmas = zeros(Float64, Nx, Ny, Nsheets)
    sheet = 1
    
    if include_surfaces
        sigmas[:,:,sheet] .= -spins[:,:,1]
        sheet += 1
    end
    
    if include_interfaces && Nz >= 2
        for i in 1:(Nz-1)
            sigmas[:,:,sheet] .= -(spins[:,:,i+1] .- spins[:,:,i])
            sheet += 1
        end
    end
    
    if include_surfaces
        sigmas[:,:,sheet] .= +spins[:,:,Nz]
        sheet += 1
    end
    
    # FFT all sheets
    sigma_k = zeros(ComplexF64, Nx, Ny, Nsheets)
    for s in 1:Nsheets
        sigma_k[:,:,s] = fft(sigmas[:,:,s])
    end
    
    # ----- Compute initial energy -----
    # H = (1/2Nxy) Σ_k σ̃† K σ̃
    H_depol = compute_total_energy(sigma_k, K_kernel, Nx, Ny)
    
    println("Initial depolarization energy: $H_depol")
    
    return DepolMCState(
        Nx, Ny, Nz,
        spins,
        kx, ky,
        K_kernel,
        sigma_k,
        z_sheets,
        flip_map,
        eps, dz, kappa,
        H_depol
    )
end


# ============================================================================
# Compute total depolarization energy from current k-space state
# ============================================================================
"""
    compute_total_energy(sigma_k, K_kernel, Nx, Ny)

Compute total depolarization energy using quadratic form in k-space:
H = (1/2Nxy) Σ_k σ̃†(k) K(k) σ̃(k)

Note: This gives the EXACT real-space energy via Parseval's theorem.
No inverse FFT needed!
"""
function compute_total_energy(sigma_k::Array{ComplexF64,3}, 
                               K_kernel::Array{ComplexF64,4},
                               Nx::Int, Ny::Int)
    Nsheets = size(sigma_k, 3)
    energy = 0.0
    
    @inbounds for i in 1:Nx, j in 1:Ny
        # Extract vectors for this k-point
        sig = @view sigma_k[i,j,:]
        K_mat = @view K_kernel[i,j,:,:]
        
        # Quadratic form: σ† K σ
        energy += real(dot(sig, K_mat * sig))
    end
    
    return 0.5 * energy / (Nx * Ny)
end


# ============================================================================
# Compute exact ΔH for a single spin flip using k-space formula
# ============================================================================
"""
    compute_flip_energy(state, ix, iy, iz)

Compute exact energy change ΔH for flipping spin at (ix, iy, iz).

Uses the k-space quadratic form:
ΔH = (1/Nxy) Σ_k Re[δσ̃† K σ̃] + (1/2Nxy) Σ_k δσ̃† K δσ̃

where δσ̃_s(k) = Δσ_s * exp(-i k·r) is computed analytically (no FFT).

Returns: ΔH (Float64), the exact energy difference.
"""
function compute_flip_energy(state::DepolMCState, ix::Int, iy::Int, iz::Int)
    # Determine which sheets are affected
    affected = state.flip_map[iz]
    
    if isempty(affected)
        return 0.0  # No depolarization contribution (shouldn't happen)
    end
    
    # Spin flip: +1 → -1 or -1 → +1
    # Change in polarization: ΔPz = -2 * current_spin
    current_spin = state.spins[ix, iy, iz]
    DPz = -2.0 * current_spin
    
    # Build δσ̃(k) for affected sheets using analytical phase factor
    # δσ̃_s(k) = Δσ_s * exp(-i[kx*(ix-1) + ky*(iy-1)])
    # Note: FFTW uses 0-based indexing in phase, so we use ix-1
    
    delta_sigma_k = zeros(ComplexF64, state.Nx, state.Ny, length(affected))
    
    for (idx, (sheet_s, coeff)) in enumerate(affected)
        Delta_sigma_s = coeff * DPz
        
        @inbounds for i in 1:state.Nx, j in 1:state.Ny
            phase = -im * (state.kx[i] * (ix - 1) + state.ky[j] * (iy - 1))
            delta_sigma_k[i,j,idx] = Delta_sigma_s * exp(phase)
        end
    end
    
    # Compute ΔH using quadratic form
    # ΔH = (1/Nxy) Σ_k Re[δσ̃† K σ̃] + (1/2Nxy) Σ_k δσ̃† K δσ̃
    
    DH_cross = 0.0  # First term: interaction with existing field
    DH_self  = 0.0  # Second term: self-energy / field reorganization
    
    @inbounds for i in 1:state.Nx, j in 1:state.Ny
        # Extract relevant submatrix of kernel
        sheet_indices = [s for (s, _) in affected]
        K_sub = state.K_kernel[i, j, sheet_indices, sheet_indices]
        
        # Current σ̃ for affected sheets
        sig_current = state.sigma_k[i, j, sheet_indices]
        
        # δσ̃ for this k-point
        dsig = delta_sigma_k[i, j, :]
        
        # Cross term: δσ̃† K σ̃
        DH_cross += real(dot(dsig, K_sub * sig_current))
        
        # Self term: δσ̃† K δσ̃
        DH_self += real(dot(dsig, K_sub * dsig))
    end
    
    Nxy = state.Nx * state.Ny
    DH = DH_cross / Nxy + 0.5 * DH_self / Nxy
    
    return DH
end


# ============================================================================
# Accept a flip: update k-space state
# ============================================================================
"""
    accept_flip!(state, ix, iy, iz)

Accept the spin flip and update the k-space charge configuration.

Updates:
1. spins[ix, iy, iz] → -spins[ix, iy, iz]
2. σ̃_s(k) → σ̃_s(k) + δσ̃_s(k) for affected sheets
3. H_depol → H_depol + ΔH

No FFT required!
"""
function accept_flip!(state::DepolMCState, ix::Int, iy::Int, iz::Int, DH::Float64)
    # Flip the spin
    current_spin = state.spins[ix, iy, iz]
    state.spins[ix, iy, iz] = -current_spin
    
    DPz = -2.0 * current_spin
    affected = state.flip_map[iz]
    
    # Update σ̃_s(k) for each affected sheet
    @inbounds for (sheet_s, coeff) in affected
        Delta_sigma_s = coeff * DPz
        
        for i in 1:state.Nx, j in 1:state.Ny
            phase = -im * (state.kx[i] * (ix - 1) + state.ky[j] * (iy - 1))
            state.sigma_k[i,j,sheet_s] += Delta_sigma_s * exp(phase)
        end
    end
    
    # Update total energy
    state.H_depol += DH
end


# ============================================================================
# Main Monte Carlo sweep
# ============================================================================
"""
    mc_sweep!(state, beta; verbose=false)

Perform one Monte Carlo sweep over all lattice sites.

For each site:
1. Compute exact ΔH using k-space formula (no FFT)
2. Accept with Metropolis probability exp(-β ΔH)
3. If accepted, update k-space state incrementally

Returns: acceptance_rate
"""
function mc_sweep!(state::DepolMCState, beta::Float64; verbose::Bool=false)
    n_accept = 0
    n_total = state.Nx * state.Ny * state.Nz
    
    # Random order to satisfy detailed balance
    sites = shuffle([(i,j,k) for i in 1:state.Nx, j in 1:state.Ny, k in 1:state.Nz])
    
    for (ix, iy, iz) in sites
        # Compute exact energy change
        DH = compute_flip_energy(state, ix, iy, iz)
        
        # Metropolis criterion
        if DH <= 0.0 || rand() < exp(-beta * DH)
            accept_flip!(state, ix, iy, iz, DH)
            n_accept += 1
        end
    end
    
    acceptance_rate = n_accept / n_total
    
    if verbose
        println("Sweep complete: acceptance rate = $(round(acceptance_rate*100, digits=2))%, E_depol = $(state.H_depol)")
    end
    
    return acceptance_rate
end


# ============================================================================
# Example: Simple test
# ============================================================================
function run_test()
    println("="^60)
    println("Testing precompiled k-space Monte Carlo for depolarization")
    println("="^60)
    
    # Small system for testing
    Nx, Ny, Nz = 8, 8, 5
    
    # Initialize with random spins
    initial = rand([-1.0, 1.0], Nx, Ny, Nz)
    
    state = initialize_depol_mc(Nx, Ny, Nz;
        eps = 1.0,
        dz = 1.0,
        lambda = Inf,  # No screening
        include_interfaces = true,
        include_surfaces = true,
        initial_spins = initial
    )
    
    println("\nSystem info:")
    println("  Grid: $Nx × $Ny × $Nz")
    println("  Number of sheets: $(length(state.z_sheets))")
    println("  Sheet positions: $(state.z_sheets)")
    println("  Initial energy: $(state.H_depol)")
    
    # Run a few Monte Carlo sweeps
    beta = 1.0  # Inverse temperature
    n_sweeps = 10
    
    println("\nRunning $n_sweeps MC sweeps at β=$beta...")
    for sweep in 1:n_sweeps
        acc_rate = mc_sweep!(state, beta, verbose=true)
    end
    
    println("\nFinal energy: $(state.H_depol)")
    println("="^60)
end

# Uncomment to run test
# run_test()
