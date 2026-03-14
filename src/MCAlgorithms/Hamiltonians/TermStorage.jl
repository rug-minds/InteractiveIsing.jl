struct ParametersSpec{Owned, Derived, I}
    owned::Owned
    derived::Derived
    info::I
end

struct Parameters{Owned, Derived, I}
    owned::Owned
    derived::Derived
    info::I
end

struct OwnedParameters 
    kwargs
    info
end

function OwnedParameters(;info = (;), kwargs...) 
    return OwnedParameters(kwargs, info)
end

function Parameters(owned::NamedTuple, derived::NamedTuple)
    return Parameters(owned, derived)
end


struct Internal{NT}
    nt::NT
end

# struct MagField{P} <: HamiltonianTerm 
#     parameters::P
# end

# function MagField(;b = nothing, c = nothing)
#     c = OwnedSpec(type = Scalar,
#                     default = StaticValue(0),
#                     info = "Magnetic field coupling constant")(c)
            
#     b = DerivedSpec(type = AbstractArray,
#             constraint = statelike,
#             default = StateLike(ConstFill, 0),
#             info = "Local magnetic field term, with field values b_i for each spin i")(b)

#     return MagField(b, c)
# end

# function parse_hamiltonian_term(terms...)
#     #parsing code to put specs into parameters

# end