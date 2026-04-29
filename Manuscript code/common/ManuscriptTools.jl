module ManuscriptTools

using InteractiveIsing
using InteractiveIsing.Processes
using CairoMakie
using GLMakie
using Dates
using DataFrames
using FileIO
using StatsBase
using XLSX

# Compatibility for a current InteractiveIsing update where `sign` was defined
# inside the package for buffer modes, shadowing `Base.sign` in sparse adjacency
# construction.
InteractiveIsing.sign(x::Real) = Base.sign(x)

graph_array(g) = InteractiveIsing.state(g)

include("plotting.jl")
include("weights.jl")
include("algorithms.jl")
include("graph_builders.jl")
include("process_recipes.jl")
include("saving.jl")
include("parallel_runs.jl")

export newmakie, makieaxis
export weightfunc1, weightfunc2, weightfunc3, weightfunc4
export weightfunc_angle_anti, weightfunc_angle_ferro
export weightfunc_shell, weightfunc_skymion, weightfunc_xy_antiferro, weightfunc_xy_dilog_antiferro
export TrianglePulseA, BiasA, SinPulseA, LinAnealingA, LinAnealingB
export ValueLogger, Recalc, ImageCapture, DatatoDataframe
export normalize_adj_by_average_col!
export graph_array
export ManuscriptParams, update_params, derived_params, build_graph
export landau_coefficients, landau_energy, landau_second_derivative, landau_hamiltonian, apply_landau_coefficients!
export AnnealRun, PulseRun, build_anneal_process, build_pulse_process
export start_anneal!, start_pulse!, fetch_run
export save_run_outputs

end
