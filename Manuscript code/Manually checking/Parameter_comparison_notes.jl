# Parameter comparison notes for the InteractiveIsing manuscript.
#
# This file is intentionally comments-only. The goal is not to preserve every
# intermediate thought, but to keep a clean working outline for:
#
#     model definition
#     reduced parameters
#     physical regimes
#     switching mechanisms
#     experiment design

# -----------------------------------------------------------------------------
# 1. Core model
# -----------------------------------------------------------------------------
#
# The working Hamiltonian is not just a simple Ising J-h model.
#
# It is better viewed as
#
#     H = H_local + H_J + H_dep + H_field + ...
#
# where:
#
#     H_local : local polynomial Landau potential
#     H_J     : dipole-dipole interaction matrix J_ij
#     H_dep   : depolarization / Coulomb feedback
#     H_field : external driving field / pulse
#
# The main scientific goal is:
#
#     start from a simpler system
#     add frustration step by step
#     compare how the switching path changes
#     explain how memory / memristive behavior emerges through different routes
#

# -----------------------------------------------------------------------------
# 2. What the code can already tune
# -----------------------------------------------------------------------------
#
# Present ingredients already available in the code:
#
#     continuous dipole variable P_i with flexible range, e.g. [-2, 2]
#     flexible local polynomial energy landscape
#     first-order local term for defect / imprint-like bias
#     higher even-order terms for double well / multiwell / stiffness control
#     flexible J through shell, angle, anti-ferro, mixed-sign constructions
#     flexible dipole spacing through lattice constants
#     charge construction + FFT Coulomb/depolarization feedback
#     external pulse field
#     multiple dynamics choices: Metropolis / Local Langevin / Global / Block
#
# This means the main missing piece is not model richness, but a clean
# comparison language.
#

# -----------------------------------------------------------------------------
# 3. Reduced-parameter philosophy
# -----------------------------------------------------------------------------
#
# The model is currently best treated as dimensionless.
#
# Instead of forcing full physical units immediately, the clean route is:
#
#     choose a reference interaction scale
#     express the other mechanisms relative to it
#
# In the present scripts, the J normalization does:
#
#     S_J = <sum_i |J_ij|> = JIsing
#
# So when JIsing = 1, what is fixed is:
#
#     the average total interaction scale felt by one dipole
#
# not:
#
#     each single bond weight
#
# Therefore, for this project, the useful reference interaction scale is:
#
#     S_J = <sum_i |J_ij|>
#
# rather than a single raw J.
#

# -----------------------------------------------------------------------------
# 4. Main reduced quantities
# -----------------------------------------------------------------------------
#
# A. Local-potential quantities
#
# For
#
#     U(P) = a P^2 + b P^4 + c P^6 + d P^8 + e P^10
#
# define:
#
#     P0                : positive well minimum
#     Ps                : barrier / saddle point
#     DeltaF_barrier    : U(Ps) - U(P0)
#
# These are direct calculations from the local polynomial.
#
#
# B. Interaction scale
#
# Use
#
#     S_J = <sum_i |J_ij|>
#
# as the typical interaction scale.
#
# Then a useful interaction-vs-barrier measure is:
#
#     Lambda_int ~ (P0^2 * S_J) / DeltaF_barrier
#
# or equivalently:
#
#     Lambda_barrier ~ DeltaF_barrier / (P0^2 * S_J)
#
#
# C. Field / defect measures
#
# If the field-like driving scale is b_typ and a defect-like local bias scale is
# h_def_typ, then useful ratios are:
#
#     Lambda_field ~ b_typ / (P0 * S_J)
#     Theta_field  ~ (P0 * b_typ) / DeltaF_barrier
#
#     Lambda_defect ~ h_def_typ / (P0 * S_J)
#     Theta_defect  ~ (P0 * h_def_typ) / DeltaF_barrier
#
#
# D. Depolarization measures
#
# For depolarization, it is important to distinguish:
#
#     reference-state reduced scale
#
# from:
#
#     current-state / trajectory diagnostic
#
# A good reference choice is:
#
#     all dipoles at +P0
#
# and then define a typical depolarization work scale from that state.
#
# Useful ratios are:
#
#     Lambda_dep_ref ~ E_dep_ref / (P0^2 * S_J)
#     Theta_dep_ref  ~ E_dep_ref / DeltaF_barrier
#
# During an actual trajectory, depolarization should also be monitored as a
# state-dependent diagnostic, because stripe states, partial switching states,
# and rough domain states will not share one fixed depolarization scale.
#

# -----------------------------------------------------------------------------
# 5. Barrier is not enough: stiffness also matters
# -----------------------------------------------------------------------------
#
# In a continuous-P model, changing high even-order terms such as
#
#     d P^8 + e P^10
#
# does not only change DeltaF_barrier.
#
# It also changes:
#
#     curvature near the well minimum
#     large-|P| wall hardness
#     how hard it is to drag a dipole away from its local minimum
#
# Therefore two local potentials can have similar barrier heights but very
# different switching behavior.
#
# Practical lesson:
#
#     barrier is the first reduced measure
#     stiffness / curvature is the second important local measure
#
# especially when comparing different high-order even terms.
#

# -----------------------------------------------------------------------------
# 6. Two different J comparison strategies
# -----------------------------------------------------------------------------
#
# Strategy A: normalized-J comparison
#
#     normalize J so that S_J stays fixed
#
# This answers:
#
#     how does interaction structure change the physics at fixed total
#     interaction scale?
#
# Good for:
#
#     shell vs angle-dependent
#     ferro vs anti-ferro
#     different frustration patterns at fixed strength
#
#
# Strategy B: raw-range / raw-NN comparison
#
#     do not renormalize after changing NN range or interaction extent
#
# This answers:
#
#     what happens when the total interaction range / budget itself changes?
#
# Good for:
#
#     NN = 1, 2, 3, ...
#     short-range vs longer-range coupling extent
#
# These two strategies are both valid, but they answer different questions and
# should not be mixed casually.
#

# -----------------------------------------------------------------------------
# 7. Input parameters vs derived quantities
# -----------------------------------------------------------------------------
#
# Input / control parameters:
#
#     a,b,c,d,e
#     defect amplitude / defect map
#     external field amplitude and pulse protocol
#     J construction rule
#     scaling and screening for depolarization
#
# Derived reduced quantities:
#
#     P0
#     DeltaF_barrier
#     S_J
#     Lambda_int
#     Lambda_barrier
#     field and defect reduced ratios
#     depolarization reduced ratios
#
# This distinction is important:
#
#     the inputs are what is swept
#     the reduced quantities are what is interpreted physically
#

# -----------------------------------------------------------------------------
# 8. Mean / median / max
# -----------------------------------------------------------------------------
#
# For local state-dependent quantities such as depolarization or local
# interaction contributions, one often gets a distribution over dipoles rather
# than one number.
#
# Then:
#
#     mean or median  -> good "typical" summary
#     max             -> useful extreme bound, but usually too sensitive to
#                        surfaces, defects, and rare outliers
#
# So for most reduced summaries:
#
#     mean / median should be primary
#     max should be secondary
#

# -----------------------------------------------------------------------------
# 9. Regime guide
# -----------------------------------------------------------------------------
#
# A. Distributed switching regime
#
# Phenomenology:
#
#     many dipoles soften and flip in a distributed way
#     no clean wall motion
#     rough nucleation / bulk-like switching
#
# Parameter tendencies:
#
#     poly/barrier small or moderate
#     J moderate but not strongly wall-forming
#     depolarization moderate or strong
#     uniform field is sufficient
#     small disorder often enhances it
#
#
# B. Clean domain-wall-motion regime
#
# Phenomenology:
#
#     an interface advances clearly
#     bulk remains relatively stable away from the wall
#
# Parameter tendencies:
#
#     poly/barrier moderate, not too soft
#     stronger local nearest-neighbor ferro tendency
#     depolarization not too strong
#     weak and slow driving
#     little disorder
#     half-up / half-down initial state is a good probe
#
#
# C. Stripe / modulated regime
#
# Phenomenology:
#
#     system avoids a uniform monodomain state
#     layered / striped / modulated patterns appear
#
# Parameter tendencies:
#
#     moderate barrier
#     competitive J, not only simple ferro
#     clear depolarization feedback
#     field not too strong
#     only modest disorder
#
#
# D. Strong pinning / stubborn-dipole regime
#
# Phenomenology:
#
#     residual unswitched dipoles
#     long tails and strong history dependence
#
# Parameter tendencies:
#
#     larger barrier
#     moderate J
#     moderate depolarization
#     important defect/disorder contribution
#

# -----------------------------------------------------------------------------
# 10. Interpreting absence of wall motion
# -----------------------------------------------------------------------------
#
# If even a manually prepared half-up / half-down state does not evolve through
# clean interface motion, but instead shows internal random or distributed
# flipping, then the current parameter regime is likely not wall-dominated.
#
# In that case:
#
#     the selected switching channel is distributed bulk-like switching
#
# rather than:
#
#     coherent domain-wall advance
#
# This is not a failed test. It is itself a mechanism diagnosis.
#

# -----------------------------------------------------------------------------
# 11. Practical experiment plan
# -----------------------------------------------------------------------------
#
# A useful minimal manuscript matrix is:
#
#     1. baseline double-well + ferro-like interaction
#     2. local landscape variation: multiwell / disorder / defect
#     3. interaction frustration variation
#     4. depolarization variation
#     5. dynamics comparison
#
# For wall questions, include a direct probe:
#
#     manually prepare half-up / half-down
#
# and test whether the interface moves.
#
# For stripe questions, use:
#
#     reference reduced depolarization scale
#     plus actual state-dependent depolarization diagnostics
#
# to separate:
#
#     what the ordered state would cost
#
# from:
#
#     what the system actually pays along the chosen trajectory

# -----------------------------------------------------------------------------
# 12. Paper 1 framing
# -----------------------------------------------------------------------------
#
# Suggested title direction:
#
#     Frustration-induced memory in ferroelectric thin films
#
# Core message:
#
#     ferroelectric memory does not emerge from a single mechanism
#     instead, distinct forms of frustration generate distinct memory pathways
#
# The real value of the present framework is not:
#
#     "we made a ferroic simulator"
#
# but:
#
#     "we built a generalized lattice framework that allows distinct frustration
#      mechanisms to be introduced and studied independently"
#
# So the scientific question for Paper 1 is:
#
#     how do different frustration mechanisms produce metastable memory states?
#
# Paper 1 should focus only on:
#
#     local landscape frustration
#     interaction frustration
#     electrostatic / depolarization frustration
#
# with only light discussion of:
#
#     relaxation
#
# and should avoid trying to fully include:
#
#     mobile defects
#     proposal dependence
#     large kinetics comparisons
#     overloaded memristor / neuromorphic packaging
#

# -----------------------------------------------------------------------------
# 13. Paper 1 outline
# -----------------------------------------------------------------------------
# ABSTRACT:
# Competing interactions, defects and boundary constraints are at the origin of frustration in materials.
# Ferroic materials are highly non-linear systems with intrinsic memory and, thus, 
# interesting as devices that combine data storage and processing. 

# In thin films of ferroic materials, depolarization/demagnetization becomes an 
# important cause of frustration and gives rise to short term memory, 
# which of much interest in neuromorphic and physics-based computing.

# Here we present a platform to simulate frustration in thin films of ferroelectrics
# In this platform we can test the behaviour of ferroelctrics films and the degree of 
# frustration when one or more of these causes are present.

# A. Introduction
#
# Key claim:
#
#     memory in ferroic systems is not tied to one microscopic origin
#     multiple frustration channels can each generate metastability and memory
#
# B. Model and framework
#
# Present:
#
#     H_local + H_J + H_dep + H_field
#
# and define the three frustration channels used in this paper.
#
# C. Reduced parameters
#
# Keep a compact reduced-parameter language:
#
#     DeltaF_barrier
#     S_J
#     Lambda_int
#     Theta_field
#     Theta_dep
#
# D. Results I: local frustration
#
#     barrier variation
#     stiffness variation
#     multiwell local landscapes
#     defect / random local bias
#
# E. Results II: interaction frustration
#
#     shell competition
#     angle dependence
#     FE/AFE competition
#     collective vs distributed switching
#
# F. Results III: electrostatic frustration
#
#     depolarization strength
#     screening
#     boundary-constrained nonuniform memory states
#
# G. Unified mechanism map
#
#     local-memory regime
#     distributed-switching regime
#     stripe / modulated regime
#     strong-pinning regime
#
# H. Conclusion
#
#     frustration pathways toward memory
#

# -----------------------------------------------------------------------------
# 14. Frustration hierarchy
# -----------------------------------------------------------------------------
#
# Level 1: local frustration
#
# Sources:
#
#     multiwell local polynomial
#     defect-like first-order terms
#     random local disorder
#
# Outcomes:
#
#     local metastability
#     analog levels
#     pinning
#
#
# Level 2: interaction frustration
#
# Sources:
#
#     competing J
#     FE/AFE competition
#     shell / angular interaction structure
#
# Outcomes:
#
#     collective switching
#     modulated or rough order
#     distributed or glassy switching pathways
#
#
# Level 3: electrostatic frustration
#
# Sources:
#
#     depolarization
#     screening
#     finite boundaries
#
# Outcomes:
#
#     suppression of monodomain order
#     domain / stripe stabilization tendency
#     nonlocal memory constraints
#
#
# Level 4: kinetic frustration
#
# Sources:
#
#     delayed field update
#     proposal dependence
#     Metropolis vs Langevin / local vs block vs global
#
# Outcomes:
#
#     slow relaxation
#     history dependence
#     nonequilibrium memory
#
# Paper 1 should cover Levels 1-3.
# Paper 2 should isolate Level 4.
#

# -----------------------------------------------------------------------------
# 15. Multi-paper roadmap
# -----------------------------------------------------------------------------
#
# Paper 1:
#
#     Frustration-induced memory in ferroelectric thin films
#
# Focus:
#
#     local + interaction + electrostatic frustration
#
# Goal:
#
#     establish the frustration hierarchy and the distinct memory pathways
#
#
# Paper 2:
#
#     Kinetic frustration and nonequilibrium memory
#
# Focus:
#
#     delayed depolarization update
#     finite electrostatic relaxation time
#     Metropolis vs Langevin
#     proposal dependence
#     block/global dynamics
#
# Goal:
#
#     show how kinetics and nonequilibrium access different memory pathways
#
#
# Paper 3:
#
#     Mobile defects / adaptive landscape
#
# Focus:
#
#     aging
#     retention
#     training
#     realistic memristive behavior
#
# Goal:
#
#     move from fixed frustration to adaptive / evolving frustration
#

# -----------------------------------------------------------------------------
# 16. Packaging warning
# -----------------------------------------------------------------------------
#
# The dangerous packaging route is:
#
#     "everything looks a bit memristive, so call it memristor behavior"
#
# This should be avoided.
#
# Safer and stronger framing:
#
#     frustration pathways toward memory
#
# because that is what the framework genuinely studies:
#
#     how metastability emerges
#     how memory emerges
#     how analog states emerge
#
# without overclaiming that every hysteretic or metastable feature is already a
# full memristor mechanism.
