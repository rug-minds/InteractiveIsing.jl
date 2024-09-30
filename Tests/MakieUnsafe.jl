using GLMakie

struct CastVec{In,Out} <: AbstractVector{Out}
    data::Vector{In}
end
Base.getindex(v::CastVec{In,Out}, i) where {In,Out} = Out(v.data[i])
Base.setindex!(v::CastVec{In,Out}, val, i) where {In,Out} = v.data[i] = In(val)
Base.length(v::CastVec) = length(v.data)
Base.size(v::CastVec) = size(v.data)
Base.eltype(v::CastVec) = eltype(v.data)
Base.IteratorSize(::Type{CastVec}) = Base.HasLength()
Base.iterate(v::CastVec, i=1) = i > length(v) ? nothing : (v[i], i+1)
CastVec(t::Type, data) = CastVec{eltype(data), t}(data)


const state = zeros(Float32, 5^2+10^3)

function update(state)
    state .= randn(5^2+10^3)
end
update(state)

# Threads.@spawn while true
#     update(state)
#     sleep(0.1)
# end


const view = @view state[5^2+1:end]
function create_unsafe_vector(view_array)
    # Get the pointer to the view array
    ptr = pointer(view_array)
    # Wrap the pointer into a Julia array without copying
    unsafe_vector = unsafe_wrap(Vector{eltype(view_array)}, ptr, length(view_array))
    return unsafe_vector
end
const unsafe_view = create_unsafe_vector(view)
const unsafeobs = Observable(unsafe_view)
const castobs = Observable(CastVec(Float64, unsafe_view))
const viewobs = Observable(view)

fig = Figure()
ax = Axis3(fig[1, 1])
display(fig)

sz = (10,10,10)
allidxs = [1:1000;]
idx2ycoord(size::NTuple{3,T}, idx) where {T} = (T(idx)-T(1)) % size[1] + T(1)
idx2xcoord(size::NTuple{3,T}, idx) where {T} = (floor(T, (idx-T(1))/size[1])) % size[2] + T(1)
idx2zcoord(size::NTuple{3,T}, idx) where {T} = floor(T, (idx-T(1))/(size[1]*size[2])) + T(1)

xs = idx2xcoord.(Ref(sz), allidxs)
ys = idx2ycoord.(Ref(sz), allidxs)
zs = idx2zcoord.(Ref(sz), allidxs)
i = meshscatter!(ax, xs, ys, zs, markersize = 0.2, color = castobs, colormap = :thermal)

#Notify timer
# Timer((timer) -> notify(unsafeobs), 0.1, interval = 1/60)

update(state)
notify(castobs)