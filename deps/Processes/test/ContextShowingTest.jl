using Test
using Processes

struct ContextShowingLongSummary end
Base.summary(io::IO, ::ContextShowingLongSummary) = print(io, "VeryLongTypeNameWithEnoughCharactersToWrap")

@testset "Context showing wraps prefixed lines to display width" begin
    buf = IOBuffer()
    io = IOContext(buf, :displaysize => (24, 20))
    Processes._print_wrapped_prefixed(
        io,
        "state = IsingGraph{Float32, β, VeryLongTypeName}",
        "│    ├── ",
        "│    │   ",
    )

    output = split(chomp(String(take!(buf))), '\n')
    @test length(output) > 1
    @test all(Base.Unicode.textwidth(line) <= 20 for line in output)
    @test occursin("β", join(output, "\n"))
end

@testset "Context showing aligns continuation guides under field branches" begin
    pc = Processes.ProcessContext(
        (;
            Metropolis_1 = Processes.SubContext(:Metropolis_1, (; model = ContextShowingLongSummary(), other = 1), (), ()),
            globals = (;),
        ),
        Processes.NameSpaceRegistry(),
    )

    buf = IOBuffer()
    io = IOContext(buf, :displaysize => (24, 34), :printcontextglobals => false)
    show(io, pc)

    output = split(chomp(String(take!(buf))), '\n')
    @test any(line -> startswith(line, "     ├── model"), output)
    @test any(line -> startswith(line, "     │   "), output)
    @test !any(line -> startswith(line, "    │   "), output)
end
