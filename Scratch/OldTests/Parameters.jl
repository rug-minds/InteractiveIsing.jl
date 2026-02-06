struct Parameters{_NT<:NamedTuple}
    _nt::_NT
end

function find_type_in_args(args, type::Symbol)
    for arg in args
        if arg isa Expr && arg.head == :(::)
            if  arg.args[2] == type
                return arg.args[1]
            end
        end
    end
    return nothing
end

function get_field_type(::Type{T}, field::Val{S}) where {T <: NamedTuple,S}
    names = fieldnames(T)
    types = fieldtypes(T)
    
    index = findfirst(==(S), names)
    
    return index === nothing ? throw(ArgumentError("Field $field not found")) : types[index]
end

function get_field_type(@specialize(nt::NamedTuple), field::Symbol)
    return get_field_type(typeof(nt), field)
end

get_field_type(nt, field::Symbol) = get_field_type(nt, Val{field}())

get_field_type(ip::Parameters{nt}, field::Symbol) where nt = get_field_type(nt, field)

Parameters(; kwargs...) = Parameters(NamedTuple(kwargs))

#Forward methods for namedtuple to Parameters
Base.getindex(p::Parameters, k::Symbol) = p._nt[k]
Base.get!(p::Parameters, k::Symbol, default) = get(p._nt, k, default)
Base.get(p::Parameters, k::Symbol) = get(p._nt, k)
Base.get(p::Parameters, k::Symbol, default) = get(p._nt, k, default)
Base.length(p::Parameters) = length(p._nt)
Base.keys(p::Parameters) = keys(p._nt)
Base.values(p::Parameters) = values(p._nt)
Base.iterate(p::Parameters, state = 1) = iterate(p._nt, state)
Base.haskey(p::Parameters, k::Symbol) = haskey(p._nt, k)

function Base.getproperty(p::Parameters{NT}, symb::Symbol) where {NT}
    return getproperty(getfield(p, :_nt), symb)
end

get_nt(p::Parameters) = getfield(p, :_nt)

# function Base.getproperty(p::Parameters, s::Symbol)
#     getproperty(p, Val{s}())
# end


const ip1 = Parameters(;a = 1, b = 2., c = :s)

function f(params)
    params.a
end

Core.Compiler.return_type(f, ip)