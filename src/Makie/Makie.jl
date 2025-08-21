import GLMakie: Axis
using GLMakie.GLFW
using GLMakie: to_native

GLMakie.activate!(;
        vsync = false,
        framerate = 60.0,
        pause_renderloop = false,
        focus_on_show = true,
        decorated = true,
        title = "Interactive Ising Simulation"
    )

include("Windows/Windows.jl")
include("WindowList.jl")
include("SimLayout.jl")

### WINDOW WindowList
### Keeps track of all open makie windows
const windowlist = WindowList()
getwindows() = windowlist
export getwindows

####
#### SimLayout REF
####

include("TimedFunctions.jl")

include("Elements/UIntTextbox.jl")

include("MakieFuncs.jl")
include("TopPanel.jl")

include("SingleView.jl")
include("MultiView.jl")
include("MidPanel.jl")

include("BottomPanel.jl")

include("BaseFig.jl")

include("SimulationWindow.jl")
include("Utils.jl")

#TODO: Overhaul this
# Sim should keep track of a vector of AbstractWindows
# Every window should be a scruct with a cleanup function
# The struct cleans up its own data and removes itself from the vector (somehow) with a uuid? (e.g. hold a dict of uuids to windows)

"""
Start the interface for the simulation displaying the graph genAdj
Or if the interface is already running, update the current view
Pass the generaating function for a particular view as the second argument
"""
function _interface(g, createview = singleView; kwargs...)
    ml = simulation[].ml[]

    # If interface is not running, create a new basefig
    if isnothing(ml[:basefig_active])
        println("Starting Interface")
        baseFig(g)
    end

    # If there is already a view, clean it up
    if !isnothing(ml[:current_view])
        cleanup(ml, ml[:current_view])
    end

    # Then assign a new one
    createview(ml, g)
end

function closeinterface()
    mlref = simulation[].ml
    deconstruct(mlref[])
    reset!(mlref)
    simulation[].ml[] = SimLayout(Figure())
end

getml() = simulation[].ml[]
export getml
export interface, closeinterface

function newwindow()
    f = Figure()
    newscreen = GLMakie.Screen()
    display(newscreen, f)
    return f, newscreen
end

## Fallback cleanup
function cleanup(ml, ::Nothing)
end

function set_colorrange(l::IsingLayer)
    midpanel(getml())[:image].colorrange[] = stateset(l)
end
export set_colorrange

