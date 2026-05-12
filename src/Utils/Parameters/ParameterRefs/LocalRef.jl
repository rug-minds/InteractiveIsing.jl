struct LocalRef <: ParameterRefBase
    name::Symbol
    indices::Tuple{Vararg{Any}}
end