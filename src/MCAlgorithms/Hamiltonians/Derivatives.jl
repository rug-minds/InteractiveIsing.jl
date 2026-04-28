abstract type BufferMode end
struct OverwriteBuffer <: BufferMode end
struct AccumulateBuffer{Sign} <: BufferMode end
AccummulateBuffer() = AccumulateBuffer{+}()
const SubtractBuffer = AccumulateBuffer{-}
sign(::AccumulateBuffer{+}) = 1
sign(::AccumulateBuffer{-}) = -1

function parameter_derivative(hterms::HamiltonianTerms, model::S; buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractIsingGraph}
    nt = named_flat_collect_broadcast(h -> parameter_derivative(h, model; buffermode), hterms)
    return nt
end




























