struct VWrapper{T}
    v::Vector{T}
end

Base.promote_rule(::Type{VWrapper{T}}, ::Type{Vector{T}}) where T = VWrapper{T}
Base.convert(::Type{VWrapper{T}}, v::Vector{T}) where T = VWrapper(v)
Base.convert(::Vector{T}, vw::Type{VWrapper{T}}, ) where T = vw.v