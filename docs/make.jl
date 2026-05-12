pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using InteractiveIsing
using InteractiveIsing.Windows

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
                "Langevin Algorithms" => "man/Langevin.md",
                "Hamiltonians" => "man/Hamiltonians.md",
                "Hamiltonian Containers" => "man/HamiltonianContainers.md",
                "Analysis" => "man/Analysis.md",
                "Processes" => "man/Processes.md",
                "Windows" => "man/Windows.md",
                "Topology" => "man/Topology.md",
            ],
            "Developer" => [
                "Registry and Scoping" => "dev/Registry.md",
                "Scoped Algorithms" => "dev/ScopedAlgorithms.md",
                "Windows Backend" => "dev/WindowsBackend.md",
                "Windows Close Lifecycle Notes" => "dev/WindowsCloseLifecycleNotes.md",
                "Langevin Boltzmann Proof" => "dev/LangevinBoltzmannProof.md",
                "Hamiltonian Term Templates" => "dev/HamiltonianTemplates.md",
                "Context" => "dev/Context.md",
            ]
        ]
        )

deploydocs(
    repo = "github.com/rug-minds/InteractiveIsing.jl.git",
)
