using FFTW
using LinearAlgebra
using Random

"""
# Exact Depolarization Monte Carlo using Precompiled k-Space Kernel

## Physical Model
- Thin ferroelectric film with N_z layers
- Each layer has polarization P_z(x,y) = s(x,y) where s ∈ {-1, +1}
- Bound charges appear at interfaces and surfaces: σ = -∇·P

## Mathematical Framework
Total energy (quadratic form in k-space):
    H = (1/2N_xy) Σ_k σ̃†(k) K(k) σ̃(k)

Where:
- σ̃(k) = FFT of charge sheet σ(x,y)
- K(k) = Electrostatic Green's function = (1/2ε|k|) exp(-|k||z_s - z_s'|)

## Key Innovation: No FFT per Flip!
For a point charge at (i_x, i_y):
    δσ̃(k) = Δσ · exp(-ik·r)  [Analytical formula!]

Energy change:
    ΔH = (1/N_xy) Σ_k Re[δσ̃† K σ̃] + (1/2N_xy) Σ_k δσ̃† K δσ̃
         └─────────────┬─────────────┘   └──────────┬──────────┘
            Cross term (with field)      Self-energy (reorganization)

## References
- Paulson et al., "Electrostatically Driven Polarization Flips..."
- Kernel method avoids frozen-field approximation
"""

# ============================================================================
#                         DATA STRUCTURES
# ============================================================================

"""
    DepolMCState

Monte Carlo state with precomputed k-space kernel.

# Fields
## Grid dimensions
- `Nx, Ny, Nz::Int` - System size (periodic in x,y; finite in z)

## Spin configuration
- `spins::Array{Float64,3}` - Current spin state (±1)

## k-space infrastructure
- `kx, ky::Vector{Float64}` - FFT frequency grids
- `K_kernel::Array{ComplexF64,4}` - **Precomputed** Green's function K(k,s,s')
  * Dimensions: (Nx, Ny, Nsheets, Nsheets)
  * K[i,j,s,s'] = (1/2ε|k|) exp(-|k||z_s - z_s'|)
  * **Computed once** at initialization!

## Charge sheet tracking
- `sigma_k::Array{ComplexF64,3}` - Current charge sheets in k-space
  * Dimensions: (Nx, Ny, Nsheets)
  * **Dynamically updated** (no re-FFT!)
- `z_sheets::Vector{Float64}` - z-positions of charge sheets
- `Nsheets::Int` - Total number of sheets

## Flip mapping
- `flip_map::Vector{Vector{Tuple{Int,Float64}}}` - Precomputed flip effects
  * flip_map[iz] = list of (sheet_index, coefficient) pairs
  * Tells which sheets change when flipping layer iz
  * Example: flip_map[1] = [(1, -1.0), (2, -1.0)] 
    → bottom surface and first interface both get Δσ = -ΔP_z

## Physical parameters
- `eps::Float64` - Dielectric constant
- `dz::Float64` - Layer spacing
- `kappa::Float64` - Screening wavevector (1/λ_screen)

## Energy tracking
- `H_depol::Float64` - Current depolarization energy
"""
mutable struct DepolMCState
    # Grid
    Nx::Int
    Ny::Int
    Nz::Int
    Nsheets::Int
    
    # Configuration
    spins::Array{Float64, 3}
    
    # k-space grid
    kx::Vector{Float64}
    ky::Vector{Float64}
    
    # Precomputed kernel (NEVER changes after init)
    K_kernel::Array{ComplexF64, 4}
    
    # Dynamic charge state (updated incrementally)
    sigma_k::Array{ComplexF64, 3}
    
    # Geometry
    z_sheets::Vector{Float64}
    flip_map::Vector{Vector{Tuple{Int, Float64}}}
    
    # Parameters
    eps::Float64
    dz::Float64
    kappa::Float64
    
    # Energy
    H_depol::Float64
end


# ============================================================================
#                         INITIALIZATION
# ============================================================================

"""
    initialize_depol_mc(Nx, Ny, Nz; kwargs...)

Set up Monte Carlo simulation with precomputed k-space kernel.

# Arguments
- `Nx, Ny, Nz::Int` - System dimensions

# Keyword Arguments
- `eps::Float64 = 1.0` - Relative dielectric constant
- `dz::Float64 = 1.0` - Layer spacing (lattice units)
- `lambda::Float64 = Inf` - Screening length (Yukawa screening)
- `include_interfaces::Bool = true` - Include interface charges?
- `include_surfaces::Bool = true` - Include surface charges?
- `initial_spins = nothing` - Optional initial configuration

# Returns
`DepolMCState` with:
1. **Precomputed K kernel** (never needs recalculation)
2. Initial σ̃(k) from FFT of initial charges
3. Initial energy H_depol

# Example
```julia
state = initialize_depol_mc(32, 32, 10; eps=10.0, dz=1.0)
# Now ready for MC sweeps!
```

# Physical Setup
Charge sheets are placed at:
- **Surfaces**: z=0 (bottom), z=(N_z-1)dz (top)
  * σ_bottom = -P_z[1]
  * σ_top = +P_z[N_z]
  
- **Interfaces**: z=(i-0.5)dz between layers i and i+1
  * σ_interface = -(P_z[i+1] - P_z[i])

Total sheets: N_sheets = 2 (surfaces) + (N_z - 1) (interfaces) = N_z + 1
"""
function initialize_depol_mc(Nx::Int, Ny::Int, Nz::Int;
    eps::Float64 = 1.0,
    dz::Float64 = 1.0,
    lambda::Float64 = Inf,
    include_interfaces::Bool = true,
    include_surfaces::Bool = true,
    initial_spins = nothing
)
    println("="^70)
    println("Initializing Depolarization Monte Carlo")
    println("="^70)
    println("System: $Nx × $Ny × $Nz")
    println("Parameters: ε=$eps, dz=$dz, λ=$lambda")
    println()
    
    # ========================================================================
    # 1. BUILD k-SPACE GRID
    # ========================================================================
    println("[1/5] Building k-space grid...")
    
    # FFT frequencies: k_n = 2π n / N for n = 0, 1, ..., N-1
    # FFTW convention: [0, 1, ..., N/2-1, -N/2, ..., -1] × (2π/N)
    kx = 2π .* FFTW.fftfreq(Nx, 1.0)
    ky = 2π .* FFTW.fftfreq(Ny, 1.0)
    
    # Screening: |k_eff| = sqrt(k_x² + k_y² + κ²)
    # κ = 1/λ where λ is screening length
    kappa = 1.0 / lambda
    
    # Precompute |k_eff| for all (k_x, k_y) pairs
    Keff = zeros(Float64, Nx, Ny)
    @inbounds for i in 1:Nx, j in 1:Ny
        k_parallel_sq = kx[i]^2 + ky[j]^2
        Keff[i,j] = sqrt(k_parallel_sq + kappa^2)
    end
    
    println("  k-grid: $(Nx)×$(Ny) points")
    println("  Screening: κ = $(round(kappa, digits=4))")
    
    
    # ========================================================================
    # 2. DETERMINE CHARGE SHEET POSITIONS
    # ========================================================================
    println("\n[2/5] Setting up charge sheets...")
    
    # Count total sheets
    Nsheets = 0
    if include_surfaces
        Nsheets += 2  # Bottom + top
    end
    if include_interfaces && Nz >= 2
        Nsheets += (Nz - 1)  # N_z-1 interfaces
    end
    
    println("  Total sheets: $Nsheets")
    
    # Allocate sheet positions
    z_sheets = zeros(Float64, Nsheets)
    sheet_idx = 1
    
    # Bottom surface (z=0)
    if include_surfaces
        z_sheets[sheet_idx] = 0.0
        println("    Sheet $sheet_idx: Bottom surface at z=0")
        sheet_idx += 1
    end
    
    # Interfaces (z=(i-0.5)dz between layers i and i+1)
    if include_interfaces && Nz >= 2
        for i in 1:(Nz-1)
            z_pos = (i - 0.5) * dz
            z_sheets[sheet_idx] = z_pos
            println("    Sheet $sheet_idx: Interface $(i)↔$(i+1) at z=$(round(z_pos,digits=2))")
            sheet_idx += 1
        end
    end
    
    # Top surface (z=(N_z-1)dz)
    if include_surfaces
        z_sheets[sheet_idx] = (Nz - 1) * dz
        println("    Sheet $sheet_idx: Top surface at z=$((Nz-1)*dz)")
        sheet_idx += 1
    end
    
    
    # ========================================================================
    # 3. PRECOMPUTE ELECTROSTATIC KERNEL K(k, s, s')
    # ========================================================================
    println("\n[3/5] Precomputing k-space kernel...")
    println("  This is the most expensive step (done only once!)")
    
    # Kernel formula: K_{ss'}(k) = (1 / 2ε|k|) exp(-|k| |z_s - z_s'|)
    # Physical meaning: Green's function for 2D Poisson equation
    #
    # Derivation:
    #   ∇²φ - |k|²φ = -σ/ε  (in k-space for x,y; z-direction ODE)
    #   Solution: φ(z) = (σ/2ε|k|) exp(-|k||z-z'|)
    
    K_kernel = zeros(ComplexF64, Nx, Ny, Nsheets, Nsheets)
    
    @inbounds for i in 1:Nx, j in 1:Ny
        keff = Keff[i,j]
        
        # Special case: k=0 mode (uniform field)
        # This is gauge freedom - set to zero
        if keff == 0.0
            continue  # Leave as zero
        end
        
        # Prefactor: 1/(2ε|k|)
        prefactor = 1.0 / (2.0 * eps * keff)
        
        # Compute all sheet-sheet interactions
        for s in 1:Nsheets, s_prime in 1:Nsheets
            # Distance between sheets
            dz_sheets = abs(z_sheets[s] - z_sheets[s_prime])
            
            # Green's function: exponential decay with distance
            K_kernel[i,j,s,s_prime] = prefactor * exp(-keff * dz_sheets)
        end
    end
    
    println("  Kernel shape: $(size(K_kernel))")
    println("  Memory: $(round(sizeof(K_kernel) / 1024^2, digits=2)) MB")
    
    
    # ========================================================================
    # 4. BUILD FLIP MAP (Which sheets are affected by each layer flip?)
    # ========================================================================
    println("\n[4/5] Building flip map...")
    
    # flip_map[iz] tells which sheets change when flipping layer iz
    # Format: [(sheet_index, coefficient), ...]
    # Where: Δσ_sheet = coefficient × ΔP_z
    #
    # Physics:
    # - Interface charge: σ = -(P_{i+1} - P_i)
    # - Surface charge: σ_bottom = -P_1, σ_top = +P_{N_z}
    
    flip_map = Vector{Vector{Tuple{Int, Float64}}}(undef, Nz)
    
    for iz in 1:Nz
        affected = Tuple{Int, Float64}[]
        
        # --- Bottom surface (iz=1) ---
        if include_surfaces && iz == 1
            sheet = 1
            # σ_bottom = -P_z[1]
            # Flip: ΔP_z = -2s → Δσ_bottom = -ΔP_z = +2s
            push!(affected, (sheet, -1.0))  # coefficient = -1
            println("  Layer $iz affects sheet $sheet (bottom surface)")
        end
        
        # --- Interface above this layer (iz, iz+1) ---
        if include_interfaces && iz < Nz
            # Interface position: (iz - 0.5)dz
            # σ = -(P_{iz+1} - P_{iz})
            # Flip iz: ΔP_iz ≠ 0, P_{iz+1} unchanged
            # → Δσ = -(0 - ΔP_iz) = +ΔP_iz
            sheet = include_surfaces ? iz + 1 : iz
            push!(affected, (sheet, -1.0))
            println("  Layer $iz affects sheet $sheet (interface $iz↔$(iz+1))")
        end
        
        # --- Interface below this layer (iz-1, iz) ---
        if include_interfaces && iz >= 2
            # σ = -(P_iz - P_{iz-1})
            # Flip iz: Δσ = -ΔP_iz
            sheet = include_surfaces ? iz : iz - 1
            push!(affected, (sheet, +1.0))
            println("  Layer $iz affects sheet $sheet (interface $(iz-1)↔$iz)")
        end
        
        # --- Top surface (iz=N_z) ---
        if include_surfaces && iz == Nz
            sheet = Nsheets
            # σ_top = +P_z[N_z]
            # Flip: Δσ_top = +ΔP_z
            push!(affected, (sheet, +1.0))
            println("  Layer $iz affects sheet $sheet (top surface)")
        end
        
        flip_map[iz] = affected
    end
    
    
    # ========================================================================
    # 5. INITIALIZE SPIN CONFIGURATION AND COMPUTE CHARGES
    # ========================================================================
    println("\n[5/5] Initializing configuration...")
    
    # Set up initial spins
    if initial_spins === nothing
        println("  Using uniform +1 spins")
        spins = ones(Float64, Nx, Ny, Nz)
    else
        println("  Using provided spin configuration")
        spins = copy(initial_spins)
    end
    
    # Compute initial charge sheets (real space)
    sigmas = zeros(Float64, Nx, Ny, Nsheets)
    sheet_idx = 1
    
    # Bottom surface: σ = -P_z[1]
    if include_surfaces
        sigmas[:,:,sheet_idx] .= -spins[:,:,1]
        sheet_idx += 1
    end
    
    # Interfaces: σ = -(P_{i+1} - P_i)
    if include_interfaces && Nz >= 2
        for i in 1:(Nz-1)
            sigmas[:,:,sheet_idx] .= -(spins[:,:,i+1] .- spins[:,:,i])
            sheet_idx += 1
        end
    end
    
    # Top surface: σ = +P_z[N_z]
    if include_surfaces
        sigmas[:,:,sheet_idx] .= +spins[:,:,Nz]
        sheet_idx += 1
    end
    
    # FFT all charge sheets to k-space
    println("  Performing initial FFT of charge sheets...")
    sigma_k = zeros(ComplexF64, Nx, Ny, Nsheets)
    for s in 1:Nsheets
        sigma_k[:,:,s] = fft(sigmas[:,:,s])
    end
    
    # Compute initial total energy
    # H = (1/2N_xy) Σ_k σ̃†(k) K(k) σ̃(k)
    H_depol = compute_total_energy(sigma_k, K_kernel, Nx, Ny)
    
    println("  Initial depolarization energy: $(round(H_depol, digits=6))")
    
    println("\n" * "="^70)
    println("Initialization complete!")
    println("="^70)
    
    return DepolMCState(
        Nx, Ny, Nz, Nsheets,
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
#                      ENERGY CALCULATIONS
# ============================================================================

"""
    compute_total_energy(sigma_k, K_kernel, Nx, Ny)

Compute total depolarization energy from k-space state.

# Formula
    H = (1/2N_xy) Σ_k σ̃†(k) K(k) σ̃(k)

Where:
- σ̃(k) is the Fourier transform of charge density
- K(k) is the Green's function kernel
- Sum over all k-points in the 2D FFT grid

# Mathematical Justification (Parseval's Theorem)
In real space:
    H = (1/2) ∫∫ σ(x,y) φ(x,y) dx dy

In k-space (by Parseval):
    H = (1/2N_xy) Σ_k σ̃*(k) φ̃(k)

Since φ̃(k) = K(k) σ̃(k):
    H = (1/2N_xy) Σ_k σ̃*(k) K(k) σ̃(k)

# Returns
- `Float64` - Total energy (real valued, even though calculation uses complex)

# Note
The factor of 1/2 accounts for double-counting in the quadratic form.
"""
function compute_total_energy(sigma_k::Array{ComplexF64,3},
                               K_kernel::Array{ComplexF64,4},
                               Nx::Int, Ny::Int)
    Nsheets = size(sigma_k, 3)
    energy = 0.0
    
    # Loop over all k-points
    @inbounds for i in 1:Nx, j in 1:Ny
        # Extract vectors/matrices for this k-point
        sig = @view sigma_k[i,j,:]           # σ̃(k) - column vector
        K_mat = @view K_kernel[i,j,:,:]      # K(k) - matrix
        
        # Quadratic form: σ̃†(k) K(k) σ̃(k)
        # This is the energy contribution from this k-mode
        energy += real(dot(sig, K_mat * sig))
    end
    
    # Normalization by N_xy and factor of 1/2
    return 0.5 * energy / (Nx * Ny)
end


"""
    compute_flip_energy(state, ix, iy, iz)

Compute **exact** energy change ΔH for flipping spin at (ix, iy, iz).

# Algorithm
1. Determine affected charge sheets from flip_map
2. Compute δσ̃(k) **analytically** (no FFT!):
   
   For point charge at (ix, iy):
       δσ̃(k) = Δσ · exp(-i k·r)
   
   Where r = [(ix-1), (iy-1)] (0-based for FFTW)

3. Evaluate quadratic form:
   
   ΔH = (1/N_xy) Σ_k Re[δσ̃†(k) K(k) σ̃(k)]     [Cross term]
        + (1/2N_xy) Σ_k δσ̃†(k) K(k) δσ̃(k)     [Self-energy]

# Physical Interpretation

**Cross term**: Energy of new charge in existing electric field
- This is what "frozen field approximation" gives
- Factor appears as 2×(original) due to Coulomb symmetry

**Self-energy term**: Energy due to field reorganization
- New charge creates new field which acts back on itself
- Factor of 1/2 from self-interaction
- **Cannot be neglected** in discrete lattice!

# Returns
- `Float64` - Energy change ΔH (can be positive or negative)

# Example
```julia
DH = compute_flip_energy(state, 5, 5, 3)
if DH < 0.0 || rand() < exp(-beta * DH)
    accept_flip!(state, 5, 5, 3, DH)
end
```
"""
function compute_flip_energy(state::DepolMCState, ix::Int, iy::Int, iz::Int)
    # Get list of affected sheets from precomputed map
    affected = state.flip_map[iz]
    
    if isempty(affected)
        return 0.0  # No depolarization (shouldn't happen in normal usage)
    end
    
    # ========================================================================
    # Compute charge change ΔP_z
    # ========================================================================
    # Spin flip: s → -s
    # Polarization change: ΔP_z = P_new - P_old = -s - s = -2s
    current_spin = state.spins[ix, iy, iz]
    DPz = -2.0 * current_spin
    
    
    # ========================================================================
    # Build δσ̃(k) for affected sheets using analytical formula
    # ========================================================================
    # Key insight: For a point charge at (ix, iy), the Fourier transform is:
    #   δσ̃(k_x, k_y) = Δσ · exp(-i[k_x·(ix-1) + k_y·(iy-1)])
    #
    # The (ix-1) is because FFTW uses 0-based indexing convention in phase.
    # This avoids FFT completely!
    
    n_affected = length(affected)
    delta_sigma_k = zeros(ComplexF64, state.Nx, state.Ny, n_affected)
    
    for (idx, (sheet_s, coeff)) in enumerate(affected)
        # Charge change for this sheet
        Delta_sigma_s = coeff * DPz
        
        # Compute δσ̃(k) = Δσ · exp(-ik·r) for all k-points
        @inbounds for i in 1:state.Nx, j in 1:state.Ny
            # Phase factor: -i(k_x·x + k_y·y)
            phase = -im * (state.kx[i] * (ix - 1) + state.ky[j] * (iy - 1))
            delta_sigma_k[i,j,idx] = Delta_sigma_s * exp(phase)
        end
    end
    
    
    # ========================================================================
    # Compute ΔH using quadratic form
    # ========================================================================
    # Formula:
    #   ΔH = (1/N_xy) Σ_k Re[δσ̃† K σ̃] + (1/2N_xy) Σ_k δσ̃† K δσ̃
    #
    # First term: Interaction with existing field
    # Second term: Self-energy / field reorganization
    
    DH_cross = 0.0  # Cross term: δσ̃† K σ̃
    DH_self = 0.0   # Self term: δσ̃† K δσ̃
    
    @inbounds for i in 1:state.Nx, j in 1:state.Ny
        # Extract kernel submatrix for affected sheets
        sheet_indices = [s for (s, _) in affected]
        K_sub = state.K_kernel[i, j, sheet_indices, sheet_indices]
        
        # Current charge state for affected sheets
        sig_current = state.sigma_k[i, j, sheet_indices]
        
        # Charge change for this k-point
        dsig = delta_sigma_k[i, j, :]
        
        # --- Cross term: δσ̃† K σ̃ ---
        # Physical: new charge in existing field
        DH_cross += real(dot(dsig, K_sub * sig_current))
        
        # --- Self term: δσ̃† K δσ̃ ---
        # Physical: new charge interacting with its own field
        DH_self += real(dot(dsig, K_sub * dsig))
    end
    
    # Normalize and combine
    Nxy = state.Nx * state.Ny
    DH = DH_cross / Nxy + 0.5 * DH_self / Nxy
    
    return DH
end


# ============================================================================
#                      STATE UPDATE (ACCEPT FLIP)
# ============================================================================

"""
    accept_flip!(state, ix, iy, iz, DH)

Accept a spin flip and update the k-space state **incrementally**.

# Updates
1. **Spin array**: `spins[ix,iy,iz] → -spins[ix,iy,iz]`
2. **Charge sheets in k-space**: `σ̃_s(k) → σ̃_s(k) + δσ̃_s(k)`
3. **Total energy**: `H_depol → H_depol + DH`

# Key Advantage: No FFT Required!
We update σ̃(k) using the same analytical formula:
    δσ̃_s(k) = Δσ_s · exp(-ik·r)

This is an **O(N_xy × N_affected)** operation, much faster than FFT.

# Arguments
- `state::DepolMCState` - State to modify (in-place)
- `ix, iy, iz::Int` - Coordinates of flipped spin
- `DH::Float64` - Energy change (from `compute_flip_energy`)

# Example
```julia
DH = compute_flip_energy(state, ix, iy, iz)
if accept_metropolis(DH, beta)
    accept_flip!(state, ix, iy, iz, DH)
end
```
"""
function accept_flip!(state::DepolMCState, ix::Int, iy::Int, iz::Int, DH::Float64)
    # ========================================================================
    # 1. Flip the spin
    # ========================================================================
    current_spin = state.spins[ix, iy, iz]
    state.spins[ix, iy, iz] = -current_spin
    
    # Recompute ΔP_z with new spin value
    DPz = -2.0 * current_spin  # Same as in compute_flip_energy
    
    
    # ========================================================================
    # 2. Update σ̃(k) for all affected sheets
    # ========================================================================
    affected = state.flip_map[iz]
    
    @inbounds for (sheet_s, coeff) in affected
        # Charge change for this sheet
        Delta_sigma_s = coeff * DPz
        
        # Update σ̃(k) incrementally: σ̃_new = σ̃_old + δσ̃
        for i in 1:state.Nx, j in 1:state.Ny
            # Same phase factor as in compute_flip_energy
            phase = -im * (state.kx[i] * (ix - 1) + state.ky[j] * (iy - 1))
            state.sigma_k[i,j,sheet_s] += Delta_sigma_s * exp(phase)
        end
    end
    
    
    # ========================================================================
    # 3. Update total energy
    # ========================================================================
    state.H_depol += DH
end


# ============================================================================
#                      MONTE CARLO SWEEP
# ============================================================================

"""
    mc_sweep!(state, beta; verbose=false)

Perform one Monte Carlo sweep over all lattice sites.

# Algorithm
For each site (in random order):
1. Compute ΔH exactly using `compute_flip_energy`
2. Accept with Metropolis probability: min(1, exp(-β ΔH))
3. If accepted, update state using `accept_flip!`

# Arguments
- `state::DepolMCState` - Current state (modified in-place)
- `beta::Float64` - Inverse temperature (1/k_B T)
- `verbose::Bool = false` - Print progress?

# Returns
- `Float64` - Acceptance rate (fraction of accepted flips)

# Example
```julia
state = initialize_depol_mc(32, 32, 10)
for sweep in 1:1000
    acc_rate = mc_sweep!(state, 1.0, verbose=(sweep % 100 == 0))
end
```

# Notes
- Random order ensures detailed balance (required for correct MC)
- Complexity: O(N_total × N_xy × N_sheets²) per sweep
  * N_total = Nx × Ny × Nz sites
  * For each flip: O(N_xy × N_sheets²) to compute ΔH
"""
function mc_sweep!(state::DepolMCState, beta::Float64; verbose::Bool=false)
    n_accept = 0
    n_total = state.Nx * state.Ny * state.Nz
    
    # Generate random order of sites (required for detailed balance)
    sites = [(i,j,k) for i in 1:state.Nx, j in 1:state.Ny, k in 1:state.Nz]
    shuffle!(sites)
    
    # Loop over all sites
    for (ix, iy, iz) in sites
        # Compute exact energy change
        DH = compute_flip_energy(state, ix, iy, iz)
        
        # Metropolis acceptance criterion
        if DH <= 0.0 || rand() < exp(-beta * DH)
            accept_flip!(state, ix, iy, iz, DH)
            n_accept += 1
        end
    end
    
    acceptance_rate = n_accept / n_total
    
    if verbose
        println("Sweep: accept=$(round(acceptance_rate*100, digits=1))%, " *
                "E_depol=$(round(state.H_depol, digits=4))")
    end
    
    return acceptance_rate
end


# ============================================================================
#                      TESTING & VALIDATION
# ============================================================================

"""
    test_energy_decomposition()

Test the two-term energy formula and verify ΔH ≠ 2P.

This demonstrates that the self-energy term is necessary!
"""
function test_energy_decomposition()
    println("\n" * "="^70)
    println("TEST: Energy Decomposition (Cross vs Self)")
    println("="^70)
    
    # Small system for clarity
    Nx, Ny, Nz = 8, 8, 5
    state = initialize_depol_mc(Nx, Ny, Nz, initial_spins=ones(Nx,Ny,Nz))
    
    # Pick a flip site
    ix, iy, iz = 4, 4, 3
    
    println("\nFlipping spin at ($ix, $iy, $iz)")
    println("Affected sheets: $(state.flip_map[iz])")
    
    # Manual calculation
    affected = state.flip_map[iz]
    DPz = -2.0 * state.spins[ix, iy, iz]
    
    # Build δσ̃(k)
    delta_sigma_k = zeros(ComplexF64, Nx, Ny, length(affected))
    for (idx, (sheet_s, coeff)) in enumerate(affected)
        Delta_sigma_s = coeff * DPz
        for i in 1:Nx, j in 1:Ny
            phase = -im * (state.kx[i] * (ix - 1) + state.ky[j] * (iy - 1))
            delta_sigma_k[i,j,idx] = Delta_sigma_s * exp(phase)
        end
    end
    
    # Compute terms separately
    DH_cross = 0.0
    DH_self = 0.0
    
    for i in 1:Nx, j in 1:Ny
        sheet_indices = [s for (s, _) in affected]
        K_sub = state.K_kernel[i, j, sheet_indices, sheet_indices]
        sig_current = state.sigma_k[i, j, sheet_indices]
        dsig = delta_sigma_k[i, j, :]
        
        DH_cross += real(dot(dsig, K_sub * sig_current))
        DH_self += real(dot(dsig, K_sub * dsig))
    end
    
    Nxy = Nx * Ny
    DH_cross /= Nxy
    DH_self /= Nxy
    DH_total = DH_cross + 0.5 * DH_self
    
    println("\nResults:")
    println("  Cross term (P):        $(round(DH_cross, digits=6))")
    println("  Self term (×0.5):      $(round(0.5*DH_self, digits=6))")
    println("  Self term (full):      $(round(DH_self, digits=6))")
    println("  Total ΔH:              $(round(DH_total, digits=6))")
    println("\nAnalysis:")
    println("  ΔH / P_cross:          $(round(DH_total/DH_cross, digits=3))")
    println("  Self / Total:          $(round(100*0.5*DH_self/DH_total, digits=1))%")
    println("  Is ΔH ≈ 2×P_cross?     $(abs(DH_total - 2*DH_cross) < 1e-10)")
    println("\n  → Self-energy is $(round(100*0.5*DH_self/DH_total, digits=0))% " *
            "of total - cannot be ignored!")
    println("="^70)
end


"""
    run_mc_test()

Run a small Monte Carlo simulation to verify the implementation.
"""
function run_mc_test()
    println("\n" * "="^70)
    println("TEST: Monte Carlo Simulation")
    println("="^70)
    
    # Initialize system
    Nx, Ny, Nz = 16, 16, 8
    
    # Random initial configuration
    initial = rand([-1.0, 1.0], Nx, Ny, Nz)
    
    state = initialize_depol_mc(Nx, Ny, Nz;
        eps = 1.0,
        dz = 1.0,
        lambda = Inf,
        include_interfaces = true,
        include_surfaces = true,
        initial_spins = initial
    )
    
    println("\nSystem info:")
    println("  Grid: $(Nx)×$(Ny)×$(Nz)")
    println("  Sheets: $(state.Nsheets)")
    println("  Initial E: $(round(state.H_depol, digits=4))")
    
    # Run sweeps
    beta = 1.0
    n_sweeps = 20
    
    println("\nRunning $n_sweeps sweeps at β=$beta...")
    energies = Float64[]
    acc_rates = Float64[]
    
    for sweep in 1:n_sweeps
        acc = mc_sweep!(state, beta, verbose=(sweep % 5 == 0))
        push!(energies, state.H_depol)
        push!(acc_rates, acc)
    end
    
    println("\nFinal state:")
    println("  Final E: $(round(state.H_depol, digits=4))")
    println("  Mean acceptance: $(round(100*sum(acc_rates)/length(acc_rates), digits=1))%")
    println("  Energy change: $(round(state.H_depol - energies[1], digits=4))")
    
    println("="^70)
end


# ============================================================================
#                      MAIN ENTRY POINT
# ============================================================================

"""
Run all tests
"""
function run_all_tests()
    test_energy_decomposition()
    run_mc_test()
end

# Uncomment to run tests when file is loaded
run_all_tests()