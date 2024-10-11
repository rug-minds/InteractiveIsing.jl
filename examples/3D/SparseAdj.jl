using SparseArrays
struct UndirectedSparse{Tv,Ti} <: SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}
    sp::SparseMatrixCSC{Tv,Ti}
    inptr::Vector{Ti}
end

inidx(s::UndirectedSparse) = s.inidx
inptr(s::UndirectedSparse) = s.inptr

Base.size(s::UndirectedSparse) = size(s.sp)
# function Base.getindex(s::UndirectedSparse, i::Int, j::Int)
#     nzptr = 1
# end
Base.setindex!(A::UndirectedSparse, _v, _i::Integer, _j::Integer) = SparseArrays._setindex_scalar!(A, _v, _i, _j, true)
function SparseArrays._setindex_scalar!(A::UndirectedSparse{Tv,Ti}, _v, _i::Integer, _j::Integer, first) where {Tv,Ti<:Integer}
    v = convert(Tv, _v)
    i = convert(Ti, _i)
    j = convert(Ti, _j)
    if !((1 <= i <= size(A, 1)) & (1 <= j <= size(A, 2)))
        throw(BoundsError(A, (i,j)))
    end
    coljfirstk = Int(SparseArrays.getcolptr(A)[j])
    coljlastk = Int(SparseArrays.getcolptr(A)[j+1] - 1)
    searchk = searchsortedfirst(rowvals(A), i, coljfirstk, coljlastk, Base.Order.Forward)
    if searchk <= coljlastk && rowvals(A)[searchk] == i
        # Column j contains entry A[i,j]. Update and return
        nonzeros(A)[searchk] = v
        nonzeros(A)[inptr(A)[searchk]] = v
        return A
    end
    # Column j does not contain entry A[i,j]. If v is nonzero, insert entry A[i,j] = v
    # and return. If to the contrary v is zero, then simply return.
    if v isa AbstractArray || v !== zero(eltype(A)) # stricter than iszero to support A[i] = -0.0
        nz = SparseArrays.getcolptr(A)[size(A, 2)+1]
        # throw exception before state is partially modified
        !isbitstype(Ti) || nz < typemax(Ti) ||
            throw(ArgumentError("nnz(A) going to exceed typemax(Ti) = $(typemax(Ti))"))

        # if nnz(A) < length(rowval/nzval): no need to grow rowval and preserve values
        SparseArrays._insert!(rowvals(A), searchk, i, nz)
        SparseArrays._insert!(nonzeros(A), searchk, v, nz)
        println("DOING THIS SHIT")
        SparseArrays._insert!(inptr(A), searchk, nz, nz)

        @simd for m in (j + 1):(size(A, 2) + 1)
            @inbounds SparseArrays.getcolptr(A)[m] += Ti(1)
        end
    end
    if first && i != j
        SparseArrays._setindex_scalar!(A, v, j, i, false)
    end
    return A
end
Base.eltype(s::UndirectedSparse) = eltype(s.sp)
SparseArrays.issparse(s::UndirectedSparse) = true
SparseArrays.nnz(s::UndirectedSparse) = nnz(s.sp)
SparseArrays.getcolptr(s::UndirectedSparse) = SparseArrays.getcolptr(s.sp)
SparseArrays.getrowval(s::UndirectedSparse) = SparseArrays.getrowval(s.sp)
SparseArrays.getnzval(s::UndirectedSparse) = SparseArrays.getnzval(s.sp)
SparseArrays.nonzeros(s::UndirectedSparse) = SparseArrays.nonzeros(s.sp)
SparseArrays.rowvals(s::UndirectedSparse) = SparseArrays.rowvals(s.sp)
Base.getproperty(s::UndirectedSparse, sym::Symbol) = sym === :colptr ? getcolptr(s.sp) : 
                                                    sym === :rowval ? getrowval(s.sp) :
                                                    sym === :nzval ? nonzeros(s.sp) :
                                                    getfield(s, sym)



struct InPtrFillTracker
    colptr::Vector{Int}
    idxs::Vector{Int}
end

slicestart(r::InPtrFillTracker, num) = r.ranges[num][1]

function UndirectedSparse(sp::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    inptr = Vector{Int}(undef, nnz(sp))
    # ranges = map((t) -> t[1]:t[2]-1, zip((@view sp.colptr[1:end-1]), @view(sp.colptr[2:end])))
    # filltracker = InPtrFillTracker(ranges, ones(Int, length(ranges)))
    filltracker = zeros(Int, size(sp,1))
    for (ptr, rowval) in enumerate(sp.rowval)
        start = sp.colptr[rowval]
        inptr[start + filltracker[rowval]] = ptr
        filltracker[rowval] += 1
    end
    return UndirectedSparse(deepcopy(sp), inptr)
end

#Symmetric sparse
s = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [2, 3, 4, 5, 1, 5, 1, 2, 3, 4], [1, 2, 3, 4, 5, 5, 1 ,2, 3, 4])
#Random sparse
us = UndirectedSparse(s)