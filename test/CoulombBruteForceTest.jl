using Test
using InteractiveIsing
using LinearAlgebra
using Statistics

const COULOMB_BRUTE_WG = @WG (;dr) -> dr == 1 ? 1.0 : 0.0 NN = 1

function brute_force_coulomb_potential(h)
    Nx, Ny, Nz = size(h.ρ)
    ax2 = h.ax^2
    ay2 = h.ay^2
    az2 = h.az^2
    cx = az2 / ax2
    cy = az2 / ay2
    li = LinearIndices((Nx, Ny, Nz))
    n = Nx * Ny * Nz

    A = zeros(eltype(h.u), n, n)
    b = vec((-az2 / h.ϵ) .* h.ρ)

    alpha_bot = 1 - h.az / h.screen_bot
    alpha_top = 1 - h.az / h.screen_top

    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        row = li[x, y, z]
        xm = x == 1 ? Nx : x - 1
        xp = x == Nx ? 1 : x + 1
        ym = y == 1 ? Ny : y - 1
        yp = y == Ny ? 1 : y + 1

        z_diag = z == 1 ? alpha_bot - 2 : z == Nz ? alpha_top - 2 : -2
        A[row, row] = z_diag - 2cx - 2cy
        A[row, li[xm, y, z]] += cx
        A[row, li[xp, y, z]] += cx
        A[row, li[x, ym, z]] += cy
        A[row, li[x, yp, z]] += cy

        z > 1 && (A[row, li[x, y, z - 1]] = 1)
        z < Nz && (A[row, li[x, y, z + 1]] = 1)
    end

    if isinf(h.screen_bot) && isinf(h.screen_top)
        gauge_row = li[1, 1, 1]
        A[gauge_row, :] .= 0
        A[gauge_row, gauge_row] = 1
        b[gauge_row] = 0
    end

    u = reshape(A \ b, Nx, Ny, Nz)

    if isinf(h.screen_bot) && isinf(h.screen_top)
        u .-= mean(@view u[:, :, 1])
    end

    return u
end

@testset "Coulomb recalc brute force" begin
    g = IsingGraph(
        3,
        2,
        2,
        Continuous(),
        COULOMB_BRUTE_WG,
        LatticeConstants(1.0, 1.2, 0.7),
        StateSet(-1.5, 1.5),
        Ising(c = ConstVal(0.0), b = 0.0) + CoulombHamiltonian(recalc = 1);
        periodic = (:x, :y),
        precision = Float64,
    )

    InteractiveIsing.graphstate(g) .= [
        1.0,
        -0.5,
        0.25,
        1.25,
        -1.0,
        0.75,
        -0.25,
        0.5,
        -1.25,
        1.0,
        0.0,
        -0.75,
    ]

    h = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
    InteractiveIsing.init!(h, g)
    InteractiveIsing.recalc!(h)

    brute = brute_force_coulomb_potential(h)
    @test h.u ≈ brute atol = 1e-10 rtol = 1e-10
end
