push!(LOAD_PATH,"../src/")

using Documenter

makedocs(sitename="InteractiveIsing Documentation",
        pages = [ 
            # "Index" => "index.md",
            "General Usage" => "man/usage.md",
            "Details" => [
                "IsingGraphs" => "man/IsingGraphs.md",
                "Indexing" => "man/Indexing.md",
                "WeightGenerators" => "man/WeightGenerator.md",
                "Generating Adjacency Lists" => "man/GeneratingAdj.md",
                "Defects" => "man/Defects.md",
                "Loops" => "man/Loops.md",
                "Parameters" => "man/Parameters.md",
                "Algorithms" => "man/Algorithms.md",
                "Hamiltonians" => "man/Hamiltonians.md",
                "Analysis" => "man/Analysis.md",
                "Processes" => "man/Processes.md",
                "Windows" => "man/Windows.md",
                "Topology" => "man/Topology.md",
            ],
            "Developer" => [
                "Registry and Scoping" => "dev/Registry.md",
                "Scoped Algorithms" => "dev/IdentifiableAlgos.md",
                "Context" => "dev/Context.md",
            ]
        ]
        )

deploydocs(
    repo = "github.com/rug-minds/InteractiveIsing.jl.git",
)
