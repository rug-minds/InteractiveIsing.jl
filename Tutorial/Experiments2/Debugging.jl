# Leaned-out version of the old debugging routine.
#
# The runnable example now lives in `examples/PulseRelaxInteractiveLines.jl`.
# It removes the state snapshot/export code and replaces the final static line
# plots with live `Windows.ContextLinesPanel` plots backed by the process
# context.

include(joinpath(@__DIR__, "..", "..", "examples", "PulseRelaxInteractiveLines.jl"))
