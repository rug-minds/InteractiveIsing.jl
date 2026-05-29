ndims(lt::AbstractLayerTopology) = length(size(lt))
ndims(lt::Type{<:AbstractLayerTopology{U,DIM}}) where {U,DIM} = DIM

export sizeto

"""
    size(top)

Return the coordinate extents stored by a layer topology.
"""
Base.size(top::AbstractLayerTopology) = top.size

"""
    size(top, i)

Return the coordinate extent along axis `i` for a layer topology.
"""
Base.size(top::AbstractLayerTopology, i) = size(top)[i]

"""
    periodic(top)

Return the periodicity type encoded in a layer topology.
"""
@inline periodic(::AbstractLayerTopology{U}) where U = U

@inline periodic(lt::AbstractLayerTopology{U}, symb) where U = periodic(U(), symb)
@inline periodicaxes(lt::AbstractLayerTopology{U}) where U = periodicaxes(U(), length(size(lt)))
@inline periodicaxes(lt::Type{<:AbstractLayerTopology{U,DIM}}) where {U,DIM} = periodicaxes(U(), DIM)

"""
    sizeto(top, size)

Return `top` sized to a layer. Fully sized topologies must already match the
layer size; topology types with explicit unsized/template support should
specialize this method.
"""
function sizeto(top::T, layer_size::NTuple{D,<:Integer}) where {D,T<:AbstractLayerTopology}
    current_size = size(top)
    length(current_size) == D ||
        throw(ArgumentError("Topology dimension $(length(current_size)) does not match layer dimension $D."))
    tuple(current_size...) == tuple(Int.(layer_size)...) ||
        throw(ArgumentError("Explicit topology size $(current_size) does not match layer size $(tuple(layer_size...))."))
    return top
end

@inline @generated function whichperiodic(lt::AbstractLayerTopology)
    periodic = fill(false, ndims(lt))
    for ax in periodicaxes(lt)
        periodic[ax] = true
    end
    periodic = tuple(periodic...)
    return :($periodic)
end


"""
Walk through the coordinates of a topology, applying periodic boundary conditions if necessary.
"""
function coordwalk(lt::AbstractLayerTopology{U}, coords...) where U
    paxes = periodicaxes(lt)
    coords = unrollreplace(coords, paxes...) do c, ax_idx
        Base.setindex(c, ax_idx, (c[ax_idx] - 1) % size(lt, ax_idx) + 1)
    end

    return coords
end
export coordwalk

Base.getindex(lt::AbstractLayerTopology, i) = lt.pvecs[i]

################################
###### IMPLEMENT ###############
################################
lattice_constants(lt::AbstractLayerTopology) = error("Not implemented for AbstractLayerTopology yet")
