abstract type BufferMode end
struct OverwriteBuffer <: BufferMode end
struct AccumulateBuffer{Sign} <: BufferMode end
AccumulateBuffer() = AccumulateBuffer{+}()
const SubtractBuffer = AccumulateBuffer{-}
Base.sign(::AccumulateBuffer{+}) = 1
Base.sign(::AccumulateBuffer{-}) = -1

function parameter_derivative(hterm::HamiltonianTerm, model::S; kwargs...) where {S <: AbstractIsingGraph}
    return parameter_derivative(hterm, graphstate(model); kwargs...)
end

function parameter_derivative(hterms::HamiltonianTerms, model::S; buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractIsingGraph}
    nt = named_flat_collect_broadcast(h -> parameter_derivative(h, graphstate(model); buffermode), hamiltonians(hterms))
    return nt
end



























