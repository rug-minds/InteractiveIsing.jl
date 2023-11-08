import GLMakie: Axis
using GLMakie.GLFW
using GLMakie: to_native

include("MakieLayout.jl")

####
#### MakieLayout REF
####
const mlref = UnRef(MakieLayout(Figure()))

include("Elements/UIntTextbox.jl")

include("SliderEdits.jl")
include("MakieFuncs.jl")
include("TopPanel.jl")

include("SingleView.jl")
include("MultiView.jl")
include("MidPanel.jl")

include("BottomPanel.jl")

include("BaseFig.jl")
include("AvgWindow.jl")

"""
Start the interface for the simulation displaying the graph genAdj
Or if the interface is already running, update the current view
Pass the generaating function for a particular view as the second argument
"""
function interface(g, createview = singleView; kwargs...)
    ml = mlref[]

    # If interface is not running, create a new basefig
    if isnothing(ml["basefig_active"])
        println("Starting Interface")
        baseFig(g)
    end

    # If there is already a view, clean it up
    if !isnothing(ml["current_view"])
        cleanup(ml, ml["current_view"])
    end

    # Then assign a new one
    createview(ml, g)
end

function closeinterface()
    ml = mlref[]
    deconstruct(ml)
    reset!(mlref)
    mlref[] = MakieLayout(Figure())
end

export interface, closeinterface

