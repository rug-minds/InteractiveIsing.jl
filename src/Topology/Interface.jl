ndims(lt::AbstractLayerTopology) = length(size(lt))
ndims(lt::Type{<:AbstractLayerTopology{U,DIM}}) where {U,DIM} = DIM

@inline periodic(lt::AbstractLayerTopology{U}, symb) where U = periodic(U(), symb)
@inline periodicaxes(lt::AbstractLayerTopology{U}) where U = periodicaxes(U(), length(size(lt)))
@inline periodicaxes(lt::Type{<:AbstractLayerTopology{U,DIM}}) where {U,DIM} = periodicaxes(U(), DIM)

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