export AllToAllWeightGenerator

struct AllToAllWeightGenerator{F} <: AbstractWeightGenerator
    func::F
    funcexp::Union{String, Expr, Symbol, Function, Nothing}
    rng::Random.AbstractRNG
end

function AllToAllWeightGenerator(func = (dr, c1, c2) -> one(Float32), rng = Random.MersenneTwister(); exp = nothing)
    AllToAllWeightGenerator{typeof(func)}(func, exp, rng)
end

getNN(::AllToAllWeightGenerator) = :all
getNN(::AllToAllWeightGenerator, dims) = ntuple(_ -> :all, dims)

@inline function (wg::AllToAllWeightGenerator)(;dr::DR, c1 = nothing, c2 = nothing) where DR
    return @inline wg.func(dr, c1, c2)
end

"""
Fully connect a layer to itself, excluding self-connections.
"""
function genLayerConnections(layer::AbstractLayerData{D}, precision, wg::AllToAllWeightGenerator, nstates) where D
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    n_conns = nstates * (nstates - 1)
    sizehint!(row_idxs, n_conns)
    sizehint!(col_idxs, n_conns)
    sizehint!(weights, n_conns)

    _fillSparseVecs(layer, precision, row_idxs, col_idxs, weights, top(layer), wg)

    return row_idxs, col_idxs, weights
end

function _fillSparseVecs(layer::AbstractLayerData{D}, precision, row_idxs, col_idxs, weights, topology, wg::AllToAllWeightGenerator) where D
    pr = parentindices(layer)[1]
    LI = LinearIndices(size(layer))

    coords = Vector{Coordinate{D}}(undef, length(layer))
    for ci in CartesianIndices(size(layer))
        idx = LI[ci]
        coords[idx] = Coordinate(topology, ci)
    end

    for col_idx in eachindex(coords)
        c1 = coords[col_idx]
        wc1 = woorldcoordinate(topology, c1)
        g_col_idx = pr[col_idx]
        for row_idx in eachindex(coords)
            row_idx == col_idx && continue
            c2 = coords[row_idx]
            wc2 = woorldcoordinate(topology, c2)
            w = precision(wg.func(dist(wc1, wc2), wc1, wc2))
            (w == 0 || isnan(w)) && continue

            push!(row_idxs, Int32(pr[row_idx]))
            push!(col_idxs, Int32(g_col_idx))
            push!(weights, w)
        end
    end
    return nothing
end

"""
Fully connect two layers by iterating all pairs of layer indices.
"""
function genLayerConnections(layer1::AbstractIsingLayer{T1,D}, layer2::AbstractIsingLayer{T2,D}, wg::AllToAllWeightGenerator) where {T1,T2,D}
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    n_conns = nStates(layer1) * nStates(layer2)
    sizehint!(row_idxs, 2*n_conns)
    sizehint!(col_idxs, 2*n_conns)
    sizehint!(weights, 2*n_conns)

    _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, wg)

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end

function _fillSparseVecs(layer1::AbstractIsingLayer{T1,D}, layer2::AbstractIsingLayer{T2,D}, row_idxs, col_idxs, weights, wg::AllToAllWeightGenerator) where {T1,T2,D}
    top1 = topology(layer1)
    top2 = topology(layer2)
    pr1 = parentindices(layer1)[1]
    pr2 = parentindices(layer2)[1]
    LI1 = LinearIndices(size(layer1))
    LI2 = LinearIndices(size(layer2))

    wc2type = typeof(woorldcoordinate(top2, Coordinate(top2, first(CartesianIndices(size(layer2))))))
    wcoords2 = Vector{wc2type}(undef, nStates(layer2))
    for ci2 in CartesianIndices(size(layer2))
        idx2 = LI2[ci2]
        wcoords2[idx2] = woorldcoordinate(top2, Coordinate(top2, ci2))
    end

    for ci1 in CartesianIndices(size(layer1))
        idx1 = LI1[ci1]
        wc1 = woorldcoordinate(top1, Coordinate(top1, ci1))
        g_col_idx = pr1[idx1]

        for idx2 in eachindex(wcoords2)
            wc2 = wcoords2[idx2]
            w = Float32(wg.func(dist(wc1, wc2), wc1, wc2))
            (w == 0 || isnan(w)) && continue

            push!(row_idxs, Int32(pr2[idx2]))
            push!(col_idxs, Int32(g_col_idx))
            push!(weights, w)
        end
    end
    return nothing
end
