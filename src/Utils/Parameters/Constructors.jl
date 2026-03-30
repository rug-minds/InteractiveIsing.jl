
ScalarParam(val::Real; description = "") = ScalarParam(typeof(val), val; description = description)
ScalarParam(T::Type, val::Real; active = true, description = "") = ParamTensor(fill(convert(T,val)), convert(T,val); active, size = tuple(), description = description)

"""
Stores a homogeneous value for vector like ParamTensors
"""
function HomogeneousParam(val::Real, size::Integer...; default = val, active = true, description = "")
    @assert !isempty(size) "HomogeneousParam requires size arguments"
    return ParamTensor(fill(val), default; size, active, description = description)
end

"""
Statically defined ParamTensor that always returns the same value
"""
function StaticParam(val, size...; description = "")
    return ParamTensor(zeros(typeof(val), size...), val, active = false, description = description)
end

# From other ParamTensors
function ParamTensor(p::ParamTensor, default = nothing , active::Bool = nothing)
    isnothing(active) && (active = isactive(p))
    isnothing(default) && (default = default(p))
    return ParamTensor(p.val, default; active, description = p.description)
end


function ParamTensor(val::T, default = nothing; size = nothing, active = false, description = "") where T
    value = val
    if T <: Array #
        et = eltype(T)
        default = default == nothing ? et(1) : convert(eltype(T), default)
    else
        default = default == nothing ? T(1) : convert(T, default)
        value = fill(val)
        et = T
    end

    isnothing(size) && (size = Base.size(value))
    DIMS = length(size)
    return ParamTensor{et, default, active, typeof(value), DIMS}(value, size, description)
end
