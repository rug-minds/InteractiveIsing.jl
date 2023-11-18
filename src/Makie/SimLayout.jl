mutable struct LayoutPanel <: AbstractDict{String, Any}
    panel::GridLayout
    elements::Dict{String, Any}
end

LayoutPanel() = LayoutPanel(GridLayout(), Dict{String, Any}())
LayoutPanel(grid::GridLayout) = LayoutPanel(grid, Dict{String, Any}())
Base.getindex(p::LayoutPanel) = p.panel
Base.getindex(p::LayoutPanel, idx) = p.elements[idx]
Base.setindex!(p::LayoutPanel, val, idx) = setindex!(p.elements, val, idx)
Base.iterate(p::LayoutPanel, state = 1) = if state == 1; return ("panel" => p.panel, 2); else; iterate(p.elements, state); end
Base.length(p::LayoutPanel) = length(p.elements)+1
Base.haskey(p::LayoutPanel, key::String) = haskey(p.elements, key)
Base.keys(p::LayoutPanel) = keys(p.elements)
Base.values(p::LayoutPanel) = values(p.elements)
Base.delete!(p::LayoutPanel, key::String) = delete!(p.elements, key)
function Base.delete!(p::LayoutPanel, keys...)
    for key in keys
        delete!(p.elements, key)
    end
end

mutable struct SimLayout
    fig::Figure
    toppanel::LayoutPanel
    midpanel::LayoutPanel
    bottompanel::LayoutPanel
    etc::Dict{String, Any}
    timers::Vector{Any}
    cleanuplist::Vector{Any}
    windowlist::WindowList
end

SimLayout(f::Figure) = SimLayout(f, LayoutPanel(), LayoutPanel(), LayoutPanel(), Dict{String, Any}(), Any[], Any[], WindowList())
function deconstruct(ml::SimLayout)
    cleanup(ml, baseFig)
    cleanup.(Ref(ml), ml.cleanuplist)
    GLFW.SetWindowShouldClose(to_native(ml["screen"]), true)
    return nothing
end

fig(ml::SimLayout) = ml.fig
fig(ml::SimLayout, f) = ml.fig = f
toppanel(ml::SimLayout) = ml.toppanel
toppanel(ml::SimLayout, p) = ml.toppanel = p
midpanel(ml::SimLayout) = ml.midpanel
midpanel(ml::SimLayout, p) = ml.midpanel = p
bottompanel(ml::SimLayout) = ml.bottompanel
bottompanel(ml::SimLayout, p) = ml.bottompanel = p
etc(ml::SimLayout) = ml.etc
Base.getindex(ml::SimLayout, idx) = try ml.etc[idx]; catch; return nothing; end
Base.setindex!(ml::SimLayout, val, idx) = setindex!(ml.etc, val, idx)
Base.iterate(ml::SimLayout, state = 1) = iterate(ml.etc, state)
Base.length(ml::SimLayout) = length(ml.etc)
Base.haskey(ml::SimLayout, key::String) = haskey(ml.etc, key)
Base.keys(ml::SimLayout) = keys(ml.etc)
Base.values(ml::SimLayout) = values(ml.etc)
Base.delete!(ml::SimLayout, key::String) = delete!(ml.etc, key)
Base.get(ml::SimLayout, key::String, default) = get(ml.etc, key, default)
Base.getkey(ml::SimLayout, key::String, default) = getkey(ml.etc, key, default)

cleanView(ml) = begin
    if haskey(etc(ml), "current_view")
        cleanup(ml, etc(ml)["current_view"])
    end
end

ImageAxis(layout; kwargs...) = 
    Axis(   layout,
            xgridvisible = false,
            ygridvisible = false,
            rightspinevisible = false,
            topspinevisible = false,
            leftspinevisible = false,
            bottomspinevisible = false,
            yticklabelsvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            xticksvisible = false,
            # halign = :right,
            # valign = :top,
            xpanlock = true,
            ypanlock = true,
            xrectzoom = false,
            yrectzoom = false;
            kwargs...
    ) 
