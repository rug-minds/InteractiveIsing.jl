export interface
include("Pausing.jl")
include("Process.jl")
interface(g; overwrite = true) = simwindow(g)
