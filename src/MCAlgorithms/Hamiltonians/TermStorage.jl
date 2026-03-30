# """
# Scalars are 0 dimensional arrays, i.e. values indexed with []
# """
# const Scalar = Union{RefValue{<:Number}, AbstractArray{<:Number, 0}}
# @inline Scalar(val) = convert(Scalar, x)
# function convert(::Type{Scalar}, x)
#     if x <: Scalar
#         return x
#     elseif x isa Number
#         return Ref(x)
#     else
#         error("Cannot convert value of type $(typeof(x)) to Scalar. Expected a Real number or a Scalar.")
#     end
# end
# """
# Workflow:
#     Owned:      0. Get a val from constructor
#                 1. Check val type,
#                 2. Apply default to value, 
#                 Reconstruct:
#                 1: x -> ensure!(x, g) 

#     Derived:    0. Get a val from constructor 
#                 1. Check val type, 
#                 Reconstruct:
#                 1. x -> ensure!(x, g)
# """
# struct ParameterSpec{SpecType}
#     symb::Symbol
#     type::Type
#     default     # Default value
#     ensure!      # Ensure the value is of the correct type, convert if possible
#     constraint  # Hard constraint with error, evaluate on state: eval (val, state) -> Bool
#     validate    # Soft constraint with warning, evaluate on state     (val, state) -> Bool
#     val
#     info::String
# end
# const OwnedSpec = ParameterSpec{:Owned}
# const DerivedSpec = ParameterSpec{:Derived}

# OwnedSpec(;type, default, validation = x -> true, info = "", kwargs...) = ParameterSpec{:Owned}(;type, default, validation, info, kwargs...)
# DerivedSpec(;type, default, validation = x -> true, info = "", kwargs...) = ParameterSpec{:Derived}(;type, default, validation, info, kwargs...)

# function ParameterSpec{T}(;type, default, validation = x -> true, info = "", kwargs...) where {T} 
#     @assert length(kwargs) == 1 "Expected exactly one keyword argument for parameter specification, got $(length(kwargs))."
#     name = first(keys(kwargs))
#     val = first(values(kwargs))

#     if isnothing(val) # If no value provided, just return the spec

#     elseif val isa Type # If a type is provided, check if it's a subtype of the expected type
#                         # If so, convert the default value to that type
#         if val <: type
#             default = convert(T, default)
#             return ParameterSpec{T}(type, default, constraint, info)
#         else
#             error("Invalid type for parameter '$name'. Expected a subtype of $(type), got $val.")
#         end
#     else                # If a value is provided, check if it's of the expected type
#         if val isa type 
#             # If Owned validate immediately
#             # If Derived, we can't validate at reconstruct
#             if T == :Owned
#                 if validation(val)
#                     return ParameterSpec{T}(type, default, constraint, val, info)
#                 else
#                     error("Validation failed for parameter '$name' with value $val.")
#                 end
#             end

#         else
#             error("Invalid value for parameter '$name'. Expected a value of type $(type), got $val.")
#         end
#     end

#     return ParameterSpec{T}(type, default, constraint, val, info)
# end

# function construct(spec::ParameterSpec{T}, state) where T
#     if T == :Owned #Already validated, just return the value
#         if isnothing(spec.val)
#             return spec.default
#         return spec.val
#     elseif T == :Derived # Need to validate and construct from state
#         i
# end
# struct Parameters{Owned, Derived, I}
#     owned::Owned
#     derived::Derived
#     info::I
# end

# function keys(params::Parameters{Owned, Derived}) where {Owned, Derived}
#     return (keys(params.owned)..., keys(params.derived)...)
# end

# function Base.fieldnames(::Type{<:Parameters{Owned, Derived}}) where {Owned, Derived}
#     return (fieldnames(Owned)..., fieldnames(Derived)...)
# end
# function Base.getproperty(params::Parameters{Owned, Derived}, name::Symbol) where {Owned, Derived}
#     if hasproperty(Owned, name)
#         return getproperty(params.owned, name)
#     elseif hasproperty(Derived, name)
#         return getproperty(params.derived, name)
#     else
#         error("No parameter named $name found in Parameters, available fields: $(fieldnames(typeof(params)))")
#     end
# end

# struct OwnedParameters 
#     kwargs
#     info
# end

# function OwnedParameters(;info = (;), kwargs...) 
#     return OwnedParameters(kwargs, info)
# end

# function Parameters(owned::NamedTuple, derived::NamedTuple)
#     return Parameters(owned, derived)
# end


# struct Internal{NT}
#     nt::NT
# end

# function Base.getproperty(ht::HamiltonianTerm, name::Symbol)
#     if hasfield(ht, :parameters)
#         return getproperty(ht.parameters, name)
#     else
#         error("No parameter named $name found in HamiltonianTerm.")
#     end
# end

# struct MagField{P} <: HamiltonianTerm 
#     parameters::P
# end

# function MagField(;b = nothing, c = nothing)
#     c = OwnedSpec(; 
#             c,
#             type = Scalar,
#             ensure! = (x,g) -> Ref(convert(eltype(g), x)),
#             default = StaticValue(0),
#             info = "Magnetic field coupling constant")
            
#     b = DerivedSpec(;
#             b,
#             type = AbstractArray,
#             validation = statelike,
#             default = StateLike(ConstFill, 0),
#             info = "Local magnetic field term, with field values b_i for each spin i")

#     return MagField(b, c)
# end

# function parse_hamiltonian_parameters(terms...)
#     #parsing code to put specs into parameters
#     owned_params = (;)
#     while true # Loop until no more OwnedSpec are found
#         el, terms = type_parse(OwnedSpec, terms...; default = nothing, error = false)
#         if isnothing(el)
#             break
#         else
#             owned_params = (;owned_params..., el.symb => el)
#         end
#     end
#     derived_params = (;)
#     while true # Loop until no more DerivedSpec are found
#         el, terms = type_parse(DerivedSpec, terms...; default = nothing, error = false)
#         if isnothing(el)
#             break
#         else
#             derived_params = (;derived_params..., el.symb => el)
#         end
#     end
#     return Parameters(owned_params, derived_params, (;))
# end

# function reconstruct(ps::Parameters, g)

# end