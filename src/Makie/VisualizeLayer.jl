export visualizelayer!

struct RepeatedRowsView{V<:AbstractVector} <: AbstractMatrix{eltype(V)}
    data::V
    nrows::Int
end

Base.size(view::RepeatedRowsView) = (view.nrows, length(view.data))
Base.getindex(view::RepeatedRowsView, ::Int, j::Int) = @inbounds view.data[j]
Base.IndexStyle(::Type{<:RepeatedRowsView}) = IndexCartesian()

function visualizelayer!(parent::Union{Figure, GridLayout}, layer::AbstractIsingLayer{T, 1}, position;
    colormap = :thermal) where T
    ax = Axis(
        parent[position...],
        xrectzoom = false,
        yrectzoom = false,
        aspect = DataAspect(),
        tellheight = true,
    )
    ax.yreversed = @load_preference("makie_y_flip", default = false)

    state_obs = Observable(RepeatedRowsView(vec(state(layer)), 10))
    im = image!(ax, state_obs; colormap, fxaa = false, interpolate = false)
    im.colorrange[] = stateset(layer)
    return state_obs
end

function visualizelayer!(parent::Union{Figure, GridLayout}, layer::AbstractIsingLayer{T, 2}, position;
    colormap = :thermal) where T
    ax = Axis(
        parent[position...],
        xrectzoom = false,
        yrectzoom = false,
        aspect = DataAspect(),
        tellheight = true,
    )
    ax.yreversed = @load_preference("makie_y_flip", default = false)

    state_obs = Observable(state(layer))
    im = image!(ax, state_obs; colormap, fxaa = false, interpolate = false)
    im.colorrange[] = stateset(layer)
    return state_obs
end

function visualizelayer!(parent::Union{Figure, GridLayout}, layer::AbstractIsingLayer{T, 3}, position;
    colormap = :thermal) where T
    ax = Axis3(parent[position...], tellheight = true)

    state_obs = Observable(vec(state(layer)))
    sz = size(layer)
    allidxs = collect(1:length(state_obs[]))
    xs = idx2xcoord.(Ref(sz), allidxs)
    ys = idx2ycoord.(Ref(sz), allidxs)
    zs = idx2zcoord.(Ref(sz), allidxs)

    plt = meshscatter!(ax, xs, ys, zs; markersize = 0.3, color = state_obs, colormap)
    plt.colorrange[] = stateset(layer)
    return state_obs
end
