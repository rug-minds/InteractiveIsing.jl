import GLMakie: Axis

include("Elements/UIntTextbox.jl")

include("SliderEdits.jl")
include("MakieLayout.jl")
include("MakieFuncs.jl")
include("TopPanel.jl")

include("SingleView.jl")
include("MultiView.jl")
include("MidPanel.jl")

include("BottomPanel.jl")

include("BaseFig.jl")

"""
Start the interface for the simulation displaying the graph genAdj
Pass the generaating function for a particular view as the second argument
"""
function interface(g, view = singleView; kwargs...)
    if isnothing(ml["basefig_active"])
        println("Starting Interface")
        createBaseFig(g)
    end

    if !isnothing(ml["current_view"])
        cleanup(ml, ml["current_view"])
    end

    view(ml, g)
end
export interface

