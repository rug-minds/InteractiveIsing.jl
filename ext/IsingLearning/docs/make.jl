pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using IsingLearning

makedocs(
    sitename = "IsingLearning Documentation",
    modules = [IsingLearning],
    pages = [
        "Overview" => "index.md",
        "XOR Findings" => "xor_findings.md",
    ],
)
