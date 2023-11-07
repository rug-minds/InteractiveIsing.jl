## GAUSSIAN BERNOULLI

function ΔE_GB(g::IsingGraph{Float32}, params, oldstate, newstate, gstate, gadj, idx, gstype)
    return nothing
end

function get_params(::typeof(ΔE_GB))
    return (;σ = Vector{Float32}, μ = Vector{Float32}, b = Vector{Float32}, bare_adj = SparseMatrixCSC{Float32,Int32})
end