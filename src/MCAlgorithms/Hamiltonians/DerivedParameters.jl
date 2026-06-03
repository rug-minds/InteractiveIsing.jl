export StateLike

# LEGACY / DEPRECATED TEMPLATE INPUTS
#
# These graph-resolved value providers predate the TermTemplate parameter
# system. New Hamiltonian templates should prefer plain defaults plus generic
# ensure functions, for example:
#
#     parameter(; b,
#         default = ConstFill(0),
#         ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
#     )
#
# `StateLike(...)` and `FromGraph(...)` are kept temporarily so older front-end
# code keeps working during migration. The TermTemplate ensure path emits a
# warning when these are used as Hamiltonian parameter inputs.
abstract type DerivedParameter end

struct StateLike{T,F} <: DerivedParameter
    default_el::F
end

StateLike(T, default_el = 0) = StateLike{T, typeof(default_el)}(default_el)

# Vector(val, size...) = fill(val, size...)

function (ss::StateLike{T})(g::AbstractSpinGraph) where T
    s = state(g)
    return filltype(T, ss.default_el, size(s)...)
end

struct FromGraph{F} <: DerivedParameter 
    f::F
end

function (fg::FromGraph{F})(g::AbstractSpinGraph) where F
    return fg.f(g)
end
