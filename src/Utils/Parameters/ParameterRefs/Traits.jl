# abstract type ContractionType end
# struct SparseAdj <: ContractionType end




abstract type PRefType end

abstract type VecLike <: PRefType end
abstract type MatrixLike <: PRefType end

struct VecRef <: VecLike end
struct MatrixRef <: MatrixLike end

sparsify(::VecRef, bool) = bool ? SparseVecRef() : VecRef()
sparsify(::MatrixRef, bool) = bool ? SparseMatrixRef() : MatrixRef()

struct SparseVecRef <: VecLike end
struct SparseMatrixRef <: MatrixLike end


abstract type ContractionType end
struct SparseColumn <: ContractionType end
struct SparseContraction <: ContractionType end
struct VectorContraction <: ContractionType end

"""
Will be constant over an iteration
"""
@generated function iterateconstant(apr::ParameterRef, args)
    rt = reftype(apr, args)
    it = iterateconstant(rt)
    return :($it)
end