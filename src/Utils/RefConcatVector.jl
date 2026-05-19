export RefConcatVector, refslices, sliceranges, turbo_fill!, turbo_copyto!, turbo_map!

"""
    RefConcatVector(vecs...)
    RefConcatVector(refs::Tuple)
    RefConcatVector{T}(refs::Tuple)

An `AbstractVector` view over a fixed tuple of refs to vectors. The number of
vector refs is part of the type parameter, and indexing treats the referenced
vectors as one sequential vector without copying their storage. Writes to the
view write through to the referenced vectors.

LoopVectorization cannot treat a multi-slice `RefConcatVector` as one strided
array. Use `turbo_fill!`, `turbo_copyto!`, or `turbo_map!` when you want
slice-wise `@turbo` loops over the backing vectors.
"""
struct RefConcatVector{T,N,Refs<:Tuple{Vararg{Base.RefValue{<:AbstractVector},N}}} <: AbstractVector{T}
    refs::Refs
end

@inline _refvec(ref::Base.RefValue{<:AbstractVector}) = ref[]

function _ref_concat_eltype(refs::Tuple)
    isempty(refs) && return Union{}
    return promote_type(map(ref -> eltype(_refvec(ref)), refs)...)
end

function RefConcatVector(refs::Tuple{Vararg{Base.RefValue{<:AbstractVector},N}}) where {N}
    T = _ref_concat_eltype(refs)
    return RefConcatVector{T,N,typeof(refs)}(refs)
end

function RefConcatVector{T}(refs::Tuple{Vararg{Base.RefValue{<:AbstractVector},N}}) where {T,N}
    return RefConcatVector{T,N,typeof(refs)}(refs)
end

@inline RefConcatVector(vecs::Vararg{AbstractVector,N}) where {N} =
    RefConcatVector(map(Ref, vecs))

@inline RefConcatVector(vecs::NTuple{N,AbstractVector}) where {N} =
    RefConcatVector(map(Ref, vecs))

"""
    refslices(v::RefConcatVector)

Return the tuple of currently referenced backing vectors. Mutating any returned
vector mutates the corresponding segment of `v`.
"""
@inline refslices(v::RefConcatVector) = map(_refvec, v.refs)

"""
    sliceranges(v::RefConcatVector)

Return the global index range covered by each backing vector.
"""
function sliceranges(v::RefConcatVector)
    offset = 0
    return map(refslices(v)) do slice
        first = offset + 1
        offset += length(slice)
        first:offset
    end
end

@inline function _ref_concat_slice_for_range(v::RefConcatVector, range)
    offset = 0
    for slice in refslices(v)
        n = length(slice)
        first(range) == offset + 1 && last(range) == offset + n && return slice
        offset += n
    end
    return view(v, range)
end

Base.IndexStyle(::Type{<:RefConcatVector}) = IndexLinear()
Base.size(v::RefConcatVector) = (length(v),)
Base.length(v::RefConcatVector) = sum(ref -> length(_refvec(ref)), v.refs; init = 0)

@inline function _ref_concat_location(v::RefConcatVector, i::Integer)
    @boundscheck checkbounds(v, i)
    offset = Int(i)
    @inbounds for ref in v.refs
        vec = _refvec(ref)
        n = length(vec)
        offset <= n && return vec, offset
        offset -= n
    end
    throw(BoundsError(v, i))
end

@inline function Base.getindex(v::RefConcatVector{T}, i::Integer) where {T}
    vec, offset = _ref_concat_location(v, i)
    return @inbounds vec[offset]::T
end

@inline function Base.setindex!(v::RefConcatVector, x, i::Integer)
    vec, offset = _ref_concat_location(v, i)
    return @inbounds setindex!(vec, x, offset)
end

function Base.iterate(v::RefConcatVector, state::Int = 1)
    state > length(v) && return nothing
    return (v[state], state + 1)
end

@inline function _ref_concat_broadcast_arg(arg, range, total_length)
    if arg isa RefConcatVector
        return _ref_concat_slice_for_range(arg, range)
    elseif arg isa AbstractVector && length(arg) == total_length
        return view(arg, range)
    elseif arg isa AbstractArray && length(arg) == total_length
        return view(vec(arg), range)
    elseif arg isa Base.Broadcast.Broadcasted
        return Base.Broadcast.Broadcasted(arg.f, _ref_concat_broadcast_args(arg.args, range, total_length))
    else
        return arg
    end
end

@inline function _ref_concat_broadcast_args(args, range, total_length)
    return map(arg -> _ref_concat_broadcast_arg(arg, range, total_length), args)
end

@inline function _ref_concat_chunked_broadcast!(dest::RefConcatVector, bc::Base.Broadcast.Broadcasted)
    bc_axes = axes(bc)
    (bc_axes == () || bc_axes == axes(dest)) ||
        throw(DimensionMismatch("broadcast axes $(bc_axes) do not match RefConcatVector axes $(axes(dest))"))
    offset = 0
    for slice in refslices(dest)
        n = length(slice)
        range = (offset + 1):(offset + n)
        args = _ref_concat_broadcast_args(bc.args, range, length(dest))
        copyto!(slice, Base.Broadcast.Broadcasted(bc.f, args, axes(slice)))
        offset += n
    end
    return dest
end

function Base.copyto!(dest::RefConcatVector, src::RefConcatVector)
    axes(src) == axes(dest) || throw(DimensionMismatch("source axes $(axes(src)) do not match RefConcatVector axes $(axes(dest))"))
    if length.(refslices(dest)) == length.(refslices(src))
        for (dest_slice, src_slice) in zip(refslices(dest), refslices(src))
            copyto!(dest_slice, src_slice)
        end
        return dest
    end
    return invoke(copyto!, Tuple{RefConcatVector, AbstractVector}, dest, src)
end

function Base.copyto!(dest::RefConcatVector, src::AbstractVector)
    axes(src) == axes(dest) || throw(DimensionMismatch("source axes $(axes(src)) do not match RefConcatVector axes $(axes(dest))"))
    offset = 0
    for slice in refslices(dest)
        n = length(slice)
        copyto!(slice, 1, src, offset + 1, n)
        offset += n
    end
    return dest
end

function Base.copyto!(dest::RefConcatVector, bc::Base.Broadcast.Broadcasted)
    return _ref_concat_chunked_broadcast!(dest, Base.Broadcast.instantiate(bc))
end

function Base.copyto!(dest::RefConcatVector, bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}})
    return _ref_concat_chunked_broadcast!(dest, Base.Broadcast.instantiate(bc))
end

function Base.materialize!(dest::RefConcatVector, bc::Base.Broadcast.Broadcasted)
    return copyto!(dest, bc)
end

if isdefined(@__MODULE__, :sumsimd)
    @eval function sumsimd(v::RefConcatVector{T}) where {T}
        total = zero(T)
        for slice in refslices(v)
            total += sumsimd(slice)
        end
        return total
    end
end

@inline _only_refslice(v::RefConcatVector{<:Any,1}) = refslices(v)[1]

@inline function turbo_getindex(v::RefConcatVector, i::Integer)
    return @inbounds v[i]
end

if isdefined(@__MODULE__, :LoopVectorization)
    import LayoutPointers

    @eval begin
        """
            turbo_fill!(dest::RefConcatVector, value)

        Fill a `RefConcatVector` by running one `@turbo` loop per backing vector.
        This is the supported LoopVectorization path for multi-slice
        `RefConcatVector`s, because the whole concatenation is not one strided
        memory region.
        """
        function turbo_fill!(dest::RefConcatVector, value)
            for slice in refslices(dest)
                LoopVectorization.@turbo for i in eachindex(slice)
                    slice[i] = value
                end
            end
            return dest
        end

        """
            turbo_copyto!(dest::RefConcatVector, src::AbstractVector)

        Copy into `dest` with one `@turbo` loop per backing vector. If `src` is
        another `RefConcatVector` with matching slice boundaries, each backing
        vector pair is copied directly.
        """
        function turbo_copyto!(dest::RefConcatVector, src::AbstractVector)
            axes(src) == axes(dest) ||
                throw(DimensionMismatch("source axes $(axes(src)) do not match RefConcatVector axes $(axes(dest))"))

            offset = 0
            total_length = length(dest)
            for dest_slice in refslices(dest)
                n = length(dest_slice)
                range = (offset + 1):(offset + n)
                src_slice = _ref_concat_broadcast_arg(src, range, total_length)
                LoopVectorization.@turbo for i in eachindex(dest_slice)
                    dest_slice[i] = src_slice[i]
                end
                offset += n
            end
            return dest
        end

        """
            turbo_map!(f, dest::RefConcatVector)
            turbo_map!(f, dest::RefConcatVector, src)
            turbo_map!(f, dest::RefConcatVector, src1, src2)

        Apply `f` with one `@turbo` loop per backing vector. The zero-source
        form maps `dest` in place; the unary and binary forms read from vector
        sources with the same axes as `dest`.
        """
        function turbo_map!(f, dest::RefConcatVector)
            for dest_slice in refslices(dest)
                LoopVectorization.@turbo for i in eachindex(dest_slice)
                    dest_slice[i] = f(dest_slice[i])
                end
            end
            return dest
        end

        function turbo_map!(f, dest::RefConcatVector, src::AbstractVector)
            axes(src) == axes(dest) ||
                throw(DimensionMismatch("source axes $(axes(src)) do not match RefConcatVector axes $(axes(dest))"))

            offset = 0
            total_length = length(dest)
            for dest_slice in refslices(dest)
                n = length(dest_slice)
                range = (offset + 1):(offset + n)
                src_slice = _ref_concat_broadcast_arg(src, range, total_length)
                LoopVectorization.@turbo for i in eachindex(dest_slice)
                    dest_slice[i] = f(src_slice[i])
                end
                offset += n
            end
            return dest
        end

        function turbo_map!(f, dest::RefConcatVector, src1::AbstractVector, src2::AbstractVector)
            axes(src1) == axes(dest) ||
                throw(DimensionMismatch("first source axes $(axes(src1)) do not match RefConcatVector axes $(axes(dest))"))
            axes(src2) == axes(dest) ||
                throw(DimensionMismatch("second source axes $(axes(src2)) do not match RefConcatVector axes $(axes(dest))"))

            offset = 0
            total_length = length(dest)
            for dest_slice in refslices(dest)
                n = length(dest_slice)
                range = (offset + 1):(offset + n)
                src1_slice = _ref_concat_broadcast_arg(src1, range, total_length)
                src2_slice = _ref_concat_broadcast_arg(src2, range, total_length)
                LoopVectorization.@turbo for i in eachindex(dest_slice)
                    dest_slice[i] = f(src1_slice[i], src2_slice[i])
                end
                offset += n
            end
            return dest
        end

        LoopVectorization.check_args(::RefConcatVector) = false
        LoopVectorization.check_args(v::RefConcatVector{<:Any,1}) =
            LoopVectorization.check_args(_only_refslice(v))

        @inline LayoutPointers.memory_reference(v::RefConcatVector{<:Any,1}) =
            LayoutPointers.memory_reference(_only_refslice(v))

        @inline LayoutPointers.stridedpointer_preserve(v::RefConcatVector{<:Any,1}) =
            LayoutPointers.stridedpointer_preserve(_only_refslice(v))
    end
end
