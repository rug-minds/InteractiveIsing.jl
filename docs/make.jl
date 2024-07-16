push!(LOAD_PATH,"../src/")

using Documenter

makedocs(sitename="InteractiveIsing Documentation")

deploydocs(
    repo = "github.com/rug-minds/InteractiveIsing.jl.git",
)