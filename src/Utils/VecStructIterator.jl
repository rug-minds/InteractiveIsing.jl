""" 
If a struct has a vector of some structs, this will create an object that essentially
acts like a vector view of the field of the structs
"""
struct VecStructIterator{T} <: AbstractVector{T}
    vec::Vector{T}
    fieldname::Symbol
end

Base.getindex(vsi::VecStructIterator, idx) = getfield(vsi.vec[idx], vsi.fieldname)
Base.setindex!(vsi::VecStructIterator, val, idx) = setfield!(vsi.vec[idx], vsi.fieldname, val)
Base.iterate(vsi::VecStructIterator, state = 1) = state > length(vsi.vec) ? nothing : (getfield(vsi.vec[state], vsi.fieldname), state + 1)
Base.size(vsi::VecStructIterator) = size(vsi.vec)

function VSI(vec::Vector{T}, fieldname) where T
   return VecStructIterator{T}(vec, fieldname)
end

"""
If a struct has a vec of some structs with some accessor function to a value,
this will create an object that behaves like a vector of the values
"""
struct VecStructIteratorAccessor{T} <: AbstractVector{T}
    vec::Vector{T}
    accessor::Function
end

Base.getindex(vsia::VecStructIteratorAccessor, idx) = vsi.accessor(vsi.vec[idx])
Base.setindex!(vsia::VecStructIteratorAccessor, val, idx) = vsi.accessor(vsi.vec[idx], val)
Base.iterate(vsia::VecStructIteratorAccessor, state = 1) = state > length(vsia.vec) ? nothing : (vsia.accessor(vsia.vec[state]), state + 1)
Base.size(vsia::VecStructIteratorAccessor) = size(vsia.vec)

"""
If a struct has a vec of some structs with some accessor function to a value,
this will create an object that behaves like a vector of the values
"""
function VSIA(vec::Vector{T}, accessor) where T
   return VecStructIteratorAccessor{T}(vec, accessor)
end