import GLMakie: Axis

set_window_config!(;
    vsync = false,
    framerate = 60.0,
    pause_renderloop = false,
    focus_on_show = true,
    decorated = true,
    title = "Interactive Ising Simulation"
)

include("SliderEdits.jl")
include("MakieLayout.jl")
include("MakieFuncs.jl")
include("TopPanel.jl")

include("SingleView.jl")
include("MultiView.jl")
include("MidPanel.jl")

include("BottomPanel.jl")

include("BaseFig.jl")


