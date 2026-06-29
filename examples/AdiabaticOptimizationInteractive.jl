# AI Generated

using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using GLMakie

# Run with:
#     julia --project examples/AdiabaticOptimizationInteractive.jl
#
# Optional backend selection:
#     AO_BACKEND=cpu    julia --project examples/AdiabaticOptimizationInteractive.jl
#     AO_BACKEND=metal  julia --project examples/AdiabaticOptimizationInteractive.jl
#     AO_BACKEND=amdgpu julia --project examples/AdiabaticOptimizationInteractive.jl

"""
    adiabatic_nn_weight(; dr)

Return a unit nearest-neighbor coupling and zero for all other offsets.
"""
function adiabatic_nn_weight(; dr::R) where {R}
    return dr == 1 ? 1.0f0 : 0.0f0
end

const ADIABATIC_NN_WG = @WG adiabatic_nn_weight NN = 1

backend_name = lowercase(get(ENV, "AO_BACKEND", "cpu"))
backend =
    backend_name == "cpu" ? :cpu :
    backend_name == "metal" ? :metal :
    backend_name == "m1" ? :m1 :
    backend_name == "amd" ? :amd :
    backend_name == "amdgpu" ? :amdgpu :
    backend_name == "auto" ? :auto :
    throw(ArgumentError("AO_BACKEND must be one of cpu, metal, m1, amd, amdgpu, or auto. Got $(backend_name)."))

nx = 100
ny = 100
nz = 10
target = fill(-1.0f0, nx * ny * nz)
mask = zeros(Float32, nx * ny * nz)
cx = cld(nx, 2)
cy = cld(ny, 2)
cz = cld(nz, 2)
half_width = max(1, min(nx, ny) ÷ 12)
half_depth = max(1, nz ÷ 5)

for z in 1:nz, y in 1:ny, x in 1:nx
    in_vertical_bar = abs(x - cx) <= half_width
    in_horizontal_bar = abs(y - cy) <= half_width
    in_mid_depth = abs(z - cz) <= half_depth
    if in_mid_depth && (in_vertical_bar || in_horizontal_bar)
        idx = x + (y - 1) * nx + (z - 1) * nx * ny
        target[idx] = 1.0f0
        mask[idx] = 1.0f0
    end
end

hamiltonian = SoftplusMarginNudging(0.85f0, target, mask, 0.25f0)

g = IsingGraph(
    nx,
    ny,
    nz,
    Continuous(),
    StateSet(-1.0f0, 1.0f0),
    ADIABATIC_NN_WG,
    LatticeConstants(1.0f0, 1.0f0, 1.0f0),
    hamiltonian;
    precision = Float32,
    periodic = (:x, :y),
)
temp!(g, 0.1f0)

algorithm = InteractiveIsing.AdiabaticOptimization(
    backend = backend,
    steps = 4_000,
    dt = 0.04f0,
    pump_start = 0.0f0,
    pump_stop = 1.35f0,
    damping = 0.01f0,
    sync_every = 2,
)

g.default_algorithm = algorithm
g.addons[:interactive] = true

host = interface(
    g;
    framerate = 30,
    polling_rate = 10,
    size = (1650, 1050),
    title = "Adiabatic Optimization 100x100x10",
)

reset_button = Button(host.figure[2, 1], label = "Reset random state", height = 34)
process_ref = Ref{Any}(nothing)

process_ref[] = createProcess(g, algorithm)

InteractiveIsing.Windows.register!(host, on(reset_button.clicks) do _
    InteractiveIsing.reset!(g)
    process_ref[] = createProcess(g, algorithm)
    return nothing
end)
