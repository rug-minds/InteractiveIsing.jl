"""
Vector that allows the shuffling of indexes without moving any internal data
"""
abstract type ShuffleMode end
struct Grouped <: ShuffleMode end
struct Ungrouped <: ShuffleMode end

abstract type CallbackMode end
struct External end
struct Internal end
Base.@kwdef struct ShuffleCallback{T}
    ref::T
    type::Symbol = :External
    convertfunc::Function = identity
    push::Function = (ref, x...) -> push!(ref, x...)
    deleteat::Function = (ref, idx) -> deleteat!(ref, idx)
    insert::Function = (ref, idx, x...) -> insert!(ref, idx, x...)
    # shift::Function = (ref, insertidx, inserteditem) -> shift!(ref, insertidx, inserteditem)
end

ShuffleCallback(ref, type = :External, convertfunc = identity; kwargs...) = ShuffleCallback(;ref, type, convertfunc, kwargs...)

Base.push!(cb::ShuffleCallback, x...) = cb.push(cb.ref, cb.convertfunc.(x)...)
Base.deleteat!(cb::ShuffleCallback, idx) = cb.deleteat(cb.ref, idx)
Base.insert!(cb::ShuffleCallback, idx, x...) = cb.insert(cb.ref, idx, cb.convertfunc.(x)...)
shuffle!(cb::ShuffleCallback, oldidx, newidx) = shuffle!(cb.ref, oldidx, newidx)
isexternal(cb::ShuffleCallback) = cb.type == :External
isinternal(cb::ShuffleCallback) = cb.type == :Internal

"""
Defines a vector like object that allows the shuffling of indexes without moving any internal data
Or inserts data ordered by type, if grouped is chosen
The internal data is type stable for quick access

ShuffleVec{T1} can be coupled with other vector{T2} like objects if those object have defined the following methods:
insert!(vec, idx, val)
deleteat!(vec, idx)
shuffle!(vec, oldidx, newidx)
swap!(vec, idx1, idx2)

and define a field pushfuncs to define what object needs to be pushed

The coupled objects will be updated when the ShuffleVec is updated

Internally coupled objects will be coupled to the internal representation of the ShuffleVec
"""
mutable struct ShuffleVec{T, Mode} <: AbstractVector{T}
    data::Tuple
    idxs::Vector{Int}
    callbacks::Vector{ShuffleCallback}
    relocate::Function

    ## TODO: Define constructors for uncoupled objects
    function ShuffleVec(data::Vector{T}; relocate = (effect, cause, shift) -> effect) where T
        idxs = collect(1:length(data))
        return new{T, Grouped}(data, idxs, ShuffleCallback[], relocate)
    end

    function ShuffleVec{T}(; relocate = (effect, cause, shift) -> effect) where T
        return new{T, Grouped}(tuple(), Vector{Int}(), ShuffleCallback[], relocate)
    end
end

deleteat(t::Tuple, idx) = (t[1:idx-1]..., t[idx+1:end]...)
push(t::Tuple, vals...) = (t..., vals...)
insert(t::Tuple, idx, val) = (t[1:idx-1]..., val, t[idx:end]...)
# insert(shufflevec::ShuffleVec, idx, val) = begin
#     shufflevec.data = insert(shufflevec.data, idx, val)
#     shufflevec.idxs = insert(shufflevec.idxs, idx, length(shufflevec.data))
#     return shufflevec
# end
Base.size(sv::ShuffleVec) = (length(sv.data),)
Base.getindex(sv::ShuffleVec{T, Mode}, i::Int) where {T, Mode} = sv.data[sv.idxs[i]]::T
Base.setindex!(sv::ShuffleVec{T, Mode}, val::T, i::Int) where {T, Mode} = sv.data = (sv.data[1:sv.idxs[i]-1]..., val, sv.data[sv.idxs[i]+1:end])
Base.IndexStyle(::Type{<:ShuffleVec}) = IndexLinear()
Base.eltype(p::ShuffleVec{T, Mode}) where {T, Mode} = T
Base.length(p::ShuffleVec) = length(p.data)
Base.iterate(p::ShuffleVec, state = 1) = state > length(p) ? nothing : (p[state], state+1)
Base.isempty(p::ShuffleVec) = isempty(p.data)
unshuffled(p::ShuffleVec) = p.data
"""
Couple two vector like objects with a func that defines what to push to the other objectname
"""
function internalcouple!(p::ShuffleVec, obj , pushfunc; kwargs...) 
    push!(p.callbacks, ShuffleCallback(obj, pushfunc, type = :Internal; kwargs...))
end
function externalcouple!(p::ShuffleVec, obj , pushfunc; kwargs...) 
    push!(p.callbacks, ShuffleCallback(obj, pushfunc, type = :External; kwargs...))
end

function uncouple(p::ShuffleVec, obj)
    idx = findfirst(x -> x.ref == obj, p.callbacks)
    deleteat!(p.callbacks, idx)
end

external_callbacks(p) = @view p.callbacks[isexternal.(p.callbacks)]
internal_callbacks(p) = @view p.callbacks[isinternal.(p.callbacks)]

# function Base.deleteat!(p::ShuffleVec, i::Integer, deletefunc::Function = (vars...) -> nothing)
function Base.deleteat!(p::ShuffleVec, i::Integer)
    internal_idx = p.idxs[i]
    old_obj = p[i]
    p.data = deleteat(p.data, p.idxs[i])
    deleteat!(p.idxs, i)

    # Update idxs
    for idx in eachindex(p.idxs)
        if p.idxs[idx] > internal_idx
            p.idxs[idx] -= 1
        end
    end

    # # Call delete function for data in the shufflevec
    # # Not standard but can be easy to clean all data
    # # Note that this doesn't work well when the shufflevecs are coupled
    # # Since it is not known what data they hold and how to clean
    # # TODO: Probably should hold this in a field
    # for new_i_idx in internal_idx:length(p)
    #     data_el = p.data[new_i_idx]
    #     deletefunc(data_el, new_i_idx)
    # end

    # Call relocator func on items that are moved from front to back
    movable_objs = p.data[internal_idx:end]
    for movable_obj in movable_objs
        p.relocate(movable_obj, old_obj, -1)
    end

    deleteat!.(external_callbacks(p), i)
    deleteat!.(internal_callbacks(p), internal_idx)

    return p
end

# For do syntax
# Base.deleteat!(f::Function, p::ShuffleVec, i::Integer) = deleteat!(p, i, f)


##TODO: Shouldn't this work inversed? THIS DOESN"T WORK
# Need to update the indexes
function Base.deleteat!(p::ShuffleVec, i::AbstractVector)

    for idx in i
        deleteat!(p.data, p.idxs[idx])
    end
    return p
end

Base.push!(p::ShuffleVec{T, Grouped}, item) where T = push!(p, (i) -> item, typeof(item))
"""
Functions should be a function that takes the idx where it is inserted and returns an item
"""
function Base.push!(p::ShuffleVec{T, Grouped}, item_f::Function, item_type) where T
    insert_idx = 0
    found_idx = findlast(x -> typeof(x) == item_type, unshuffled(p))
    insert_idx = 0
    if !isnothing(found_idx)

        insert_idx = found_idx + 1
        # Update all internal indexes
        for idx in eachindex(p.idxs)
            if p.idxs[idx] >= insert_idx
                p.idxs[idx] += 1
            end
        end
        
    else # Insertidx is at the end
        insert_idx = length(p.data) + 1
    end

    new_obj = item_f(insert_idx)
    p.data = insert(p.data, insert_idx, new_obj)
    p.idxs = push!(p.idxs, insert_idx)

    # Call relocator function for all data that is moved from back to front
    # TODO: Can I make this a view?
    movable_objs = p.data[end:-1:insert_idx+1]
    for movable_obj in movable_objs
        p.relocate(movable_obj, new_obj, 1)
    end

    # Push to all coupled objects
    for callback in external_callbacks(p)
        push!(callback, new_obj)
    end

    # If coupled internally insert it
    for callback in internal_callbacks(p)
        insert!(callback, insert_idx, new_obj)
    end

    return p
end

## PUSH MULTIPLE ITEMS AT THE SAME TIME
# function Base.push!(p::ShuffleVec{T, Ungrouped}, items...) where T
#     p.data = push(p.data, items...)
#     push!(p.idxs, length(p.data))

#     for idx in eachindex(all_couples(p))
#         push!.(callbacks(p), items...)
#     end

#     return p
# end

function shuffle!(p::ShuffleVec, oldidx, newidx)
    newidxs = copy(p.idxs)
    internal_idx = p.idxs[oldidx]

    shift_right = newidx < oldidx
    block = shift_right ? (newidx:(oldidx-1)) : ((oldidx+1):newidx)

    if shift_right
        newidxs[block.+1] = p.idxs[block]
        newidxs[newidx] = internal_idx        
    else
        newidxs[block.-1] = p.idxs[block]
        newidxs[newidx] = internal_idx
    end

    p.idxs .= newidxs

    shuffle!.(external_callbacks(p), oldidx, newidx)
end

# TODO: Implement swap!

@inline internalidx(p::ShuffleVec, external_idx::Integer) = p.idxs[external_idx]
@inline externalidx(p::ShuffleVec, internal_idx::Integer) = findfirst(p.idxs .== internal_idx)

Base.convert(::Type{ShuffleVec{T}}, p::Vector{T}) where T = ShuffleVec(p)
Base.convert(::Type{Vector{T}}, p::ShuffleVec{T}) where T = p.data