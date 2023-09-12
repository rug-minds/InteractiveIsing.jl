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

mutable struct MakieLayout
    fig::Figure
    toppanel::LayoutPanel
    midpanel::LayoutPanel
    bottompanel::LayoutPanel
    etc::Dict{String, Any}
end
MakieLayout(f::Figure) = MakieLayout(f, LayoutPanel(), LayoutPanel(), LayoutPanel(), Dict{String, Any}())

fig(ml::MakieLayout) = ml.fig
fig(ml::MakieLayout, f) = ml.fig = f
toppanel(ml::MakieLayout) = ml.toppanel
toppanel(ml::MakieLayout, p) = ml.toppanel = p
midpanel(ml::MakieLayout) = ml.midpanel
midpanel(ml::MakieLayout, p) = ml.midpanel = p
bottompanel(ml::MakieLayout) = ml.bottompanel
bottompanel(ml::MakieLayout, p) = ml.bottompanel = p
etc(ml::MakieLayout) = ml.etc
Base.getindex(ml::MakieLayout, idx) = ml.etc[idx]
Base.setindex!(ml::MakieLayout, val, idx) = setindex!(ml.etc, val, idx)
Base.iterate(ml::MakieLayout, state = 1) = iterate(ml.etc, state)
Base.length(ml::MakieLayout) = length(ml.etc)
Base.haskey(ml::MakieLayout, key::String) = haskey(ml.etc, key)
Base.keys(ml::MakieLayout) = keys(ml.etc)
Base.values(ml::MakieLayout) = values(ml.etc)

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
