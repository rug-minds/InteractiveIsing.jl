using CairoMakie
using MLDatasets

"""Return the four checkerboard XOR input patterns used by local XOR runs."""
function xor_checkerboard_patterns(side::T) where {T<:Integer}
    cases = ((false, false), (false, true), (true, false), (true, true))
    patterns = Matrix{Float32}[]
    for (a, b) in cases
        pattern = Matrix{Float32}(undef, Int(side), Int(side))
        aval = a ? 1f0 : -1f0
        bval = b ? 1f0 : -1f0
        for col in axes(pattern, 2), row in axes(pattern, 1)
            pattern[row, col] = isodd(row + col) ? aval : bval
        end
        push!(patterns, pattern)
    end
    return cases, patterns
end

"""Return the four edge-input XOR line patterns."""
function xor_edge_patterns(side::T) where {T<:Integer}
    cases = ((false, false), (false, true), (true, false), (true, true))
    patterns = Matrix{Float32}[]
    for (a, b) in cases
        pattern = Matrix{Float32}(undef, Int(side), 1)
        aval = a ? 1f0 : -1f0
        bval = b ? 1f0 : -1f0
        for row in axes(pattern, 1)
            pattern[row, 1] = isodd(row) ? aval : bval
        end
        push!(patterns, pattern)
    end
    return cases, patterns
end

"""Load one MNIST example image as a Float32 matrix."""
function mnist_example_image()
    images, labels = MNIST(split = :train)[:]
    idx = findfirst(==(7), labels)
    isnothing(idx) && (idx = 1)
    return Float32.(images[:, :, idx]), Int(labels[idx])
end

"""Return a 55x55 inlaid MNIST grid with live separator sites set to zero."""
function inlaid_mnist_grid(image::A) where {A<:AbstractMatrix}
    grid = zeros(Float32, 55, 55)
    for col in axes(image, 2), row in axes(image, 1)
        grid[2 * row - 1, 2 * col - 1] = Float32(image[row, col])
    end
    return grid
end

"""Write a short note explaining the generated schematic folder."""
function write_schematic_readme!(outdir::P, title::S, files::F) where {P<:AbstractString,S<:AbstractString,F<:AbstractVector}
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# ", title)
        println(io)
        println(io, "Use of this folder: generated input-pattern and architecture schematic figures for the aggregate experiments. These are documentation figures, not training artifacts.")
        println(io)
        for file in files
            println(io, "- `", basename(file), "`")
        end
    end
    return joinpath(outdir, "README.md")
end

"""Draw a compact stack of architecture layers and arrows."""
function draw_architecture!(
    fig,
    cell,
    title::S,
    layers::L,
    arrows::A,
) where {S<:AbstractString,L<:AbstractVector,A<:AbstractVector}
    ax = Axis(fig[cell...], title = title, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    limits!(ax, -1, 13, 0, 8)

    centers = range(1.4, 10.6; length = length(layers))
    for (idx, layer) in enumerate(layers)
        x = centers[idx]
        width = Float64(layer.width)
        height = Float64(layer.height)
        left = x - width / 2
        bottom = 4 - height / 2
        poly!(ax, Point2f[(left, bottom), (left + width, bottom), (left + width, bottom + height), (left, bottom + height)];
            color = layer.color, strokecolor = :gray25, strokewidth = 2)
        text!(ax, x, bottom + height + 0.35; text = layer.name, align = (:center, :bottom), fontsize = 15)
        text!(ax, x, bottom - 0.25; text = layer.detail, align = (:center, :top), fontsize = 12)
    end
    for (idx, label) in enumerate(arrows)
        x1 = centers[idx] + Float64(layers[idx].width) / 2 + 0.15
        x2 = centers[idx + 1] - Float64(layers[idx + 1].width) / 2 - 0.15
        y = 4
        lines!(ax, [x1, x2], [y, y]; linewidth = 2.5, color = :gray25)
        scatter!(ax, [x2], [y]; marker = :rtriangle, markersize = 18, color = :gray25)
        text!(ax, (x1 + x2) / 2, y + 0.35; text = label, align = (:center, :bottom), fontsize = 12)
    end
    return ax
end

"""Save a figure showing checkerboard XOR input patterns and architecture."""
function save_xor_checkerboard_schematic(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    cases, patterns = xor_checkerboard_patterns(8)
    fig = Figure(size = (1500, 850))
    Label(fig[1, 1:4], "XOR checkerboard input encoding", fontsize = 24, tellwidth = false)
    for (idx, pattern) in enumerate(patterns)
        ax = Axis(fig[2, idx], title = "input $(Int(cases[idx][1]))$(Int(cases[idx][2]))")
        hidedecorations!(ax)
        heatmap!(ax, pattern; colormap = :balance, colorrange = (-1, 1))
    end
    layers = [
        (name = "Input", detail = "8x8 checkerboard fields", width = 1.4, height = 1.4, color = (:dodgerblue3, 0.35)),
        (name = "Hidden 1", detail = "8x8 local r1", width = 1.8, height = 1.8, color = (:seagreen, 0.35)),
        (name = "Hidden 2", detail = "4x4 local r2", width = 1.15, height = 1.15, color = (:goldenrod3, 0.35)),
        (name = "Output", detail = "4x4 target/vote", width = 1.15, height = 1.15, color = (:firebrick, 0.35)),
    ]
    draw_architecture!(fig, (3, 1:4), "Checkerboard/local XOR architecture", layers, ["r1 = 1..5", "r2 = 1..2", "r2 readout"])
    path = joinpath(outdir, "schematic.png")
    save(path, fig)
    return path
end

"""Save a figure showing the `2 -> 2x2 -> 4` majority-vote XOR baseline."""
function save_xor_majority_vote_schematic(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    fig = Figure(size = (1450, 760))
    ax = Axis(fig[1, 1], title = "Bipolar XOR inputs", xlabel = "case", ylabel = "input spin")
    hidedecorations!(ax, grid = false)
    inputs = [-1 -1 1 1; -1 1 -1 1]
    heatmap!(ax, 1:4, 1:2, inputs'; colormap = :balance, colorrange = (-1, 1))
    text!(ax, 1:4, fill(2.35, 4); text = ["00", "01", "10", "11"], align = (:center, :center), fontsize = 18)

    layers = [
        (name = "Input", detail = "2 spins", width = 0.75, height = 1.4, color = (:dodgerblue3, 0.35)),
        (name = "Hidden", detail = "2x2 spins", width = 1.2, height = 1.2, color = (:seagreen, 0.35)),
        (name = "Output", detail = "4 replicas", width = 1.2, height = 0.75, color = (:firebrick, 0.35)),
    ]
    draw_architecture!(fig, (1, 2:4), "Majority-vote baseline", layers, ["all-to-all", "all-to-all; vote by mean sign"])
    path = joinpath(outdir, "schematic.png")
    save(path, fig)
    return path
end

"""Save a figure showing edge-input XOR patterns and architecture."""
function save_xor_edge_schematic(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    cases, patterns = xor_edge_patterns(16)
    fig = Figure(size = (1500, 850))
    Label(fig[1, 1:4], "XOR edge input encoding", fontsize = 24, tellwidth = false)
    for (idx, pattern) in enumerate(patterns)
        ax = Axis(fig[2, idx], title = "input $(Int(cases[idx][1]))$(Int(cases[idx][2]))", aspect = DataAspect())
        hidedecorations!(ax)
        heatmap!(ax, pattern; colormap = :balance, colorrange = (-1, 1))
    end
    layers = [
        (name = "Input layer", detail = "separate 16x1 line", width = 0.55, height = 2.3, color = (:dodgerblue3, 0.35)),
        (name = "Dynamic layer", detail = "full 16x16 live field", width = 2.2, height = 2.2, color = (:seagreen, 0.35)),
        (name = "Output layer", detail = "separate 16x1 line", width = 0.55, height = 2.3, color = (:firebrick, 0.35)),
    ]
    draw_architecture!(fig, (3, 1:4), "Edge application architecture", layers, ["fanout to left edge", "readout from right edge"])
    path = joinpath(outdir, "schematic.png")
    save(path, fig)
    return path
end

"""Save a figure showing the single-hidden local MNIST baseline."""
function save_mnist_local_schematic(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    image, label = mnist_example_image()
    fig = Figure(size = (1500, 850))
    ax = Axis(fig[1, 1], title = "MNIST input example: digit $(label)", aspect = DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, image; colormap = :grays, colorrange = (0, 1))
    layers = [
        (name = "Input fields", detail = "28x28 image", width = 1.55, height = 1.55, color = (:dodgerblue3, 0.35)),
        (name = "Hidden", detail = "11x11 or about 120 spins", width = 1.25, height = 1.25, color = (:seagreen, 0.35)),
        (name = "Output", detail = "40 spins, 4 per digit", width = 1.0, height = 0.65, color = (:firebrick, 0.35)),
    ]
    draw_architecture!(fig, (1, 2:4), "Local MNIST baseline", layers, ["local image fanout", "dense readout"])
    path = joinpath(outdir, "schematic.png")
    save(path, fig)
    return path
end

"""Save a figure showing the two-hidden-layer CNN-style MNIST architecture."""
function save_mnist_cnn_schematic(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    image, label = mnist_example_image()
    fig = Figure(size = (1500, 850))
    ax = Axis(fig[1, 1], title = "MNIST input example: digit $(label)", aspect = DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, image; colormap = :grays, colorrange = (0, 1))
    layers = [
        (name = "Input fields", detail = "28x28 image", width = 1.55, height = 1.55, color = (:dodgerblue3, 0.35)),
        (name = "Hidden 1", detail = "28x28", width = 1.55, height = 1.55, color = (:seagreen, 0.35)),
        (name = "Hidden 2", detail = "11x11 or 14x14", width = 1.2, height = 1.2, color = (:goldenrod3, 0.35)),
        (name = "Output", detail = "40 spins, 4 per digit", width = 1.0, height = 0.65, color = (:firebrick, 0.35)),
    ]
    draw_architecture!(fig, (1, 2:4), "Two-hidden-layer local MNIST", layers, ["local radius", "local radius", "dense readout"])
    path = joinpath(outdir, "schematic.png")
    save(path, fig)
    return path
end

"""Save a figure showing the inlaid MNIST input layer and readout architecture."""
function save_mnist_inlaid_schematic(outdir::P) where {P<:AbstractString}
    mkpath(outdir)
    image, label = mnist_example_image()
    inlaid = inlaid_mnist_grid(image)
    fig = Figure(size = (1500, 850))
    ax1 = Axis(fig[1, 1], title = "Original 28x28 digit $(label)", aspect = DataAspect())
    ax2 = Axis(fig[1, 2], title = "55x55 inlaid pixels with live separator sites", aspect = DataAspect())
    hidedecorations!(ax1)
    hidedecorations!(ax2)
    heatmap!(ax1, image; colormap = :grays, colorrange = (0, 1))
    heatmap!(ax2, inlaid; colormap = :grays, colorrange = (0, 1))
    layers = [
        (name = "Inlaid input", detail = "55x55 fixed pixels + separators", width = 2.0, height = 2.0, color = (:dodgerblue3, 0.35)),
        (name = "Output", detail = "40 spins, 4 per digit", width = 1.0, height = 0.65, color = (:firebrick, 0.35)),
    ]
    draw_architecture!(fig, (1, 3:4), "Inlaid MNIST readout", layers, ["dense pixel/live readout"])
    path = joinpath(outdir, "schematic.png")
    save(path, fig)
    return path
end

"""Generate all schematic figures used by the current experiment folders."""
function main()
    root = @__DIR__
    outputs = String[]
    append!(outputs, [
        save_xor_majority_vote_schematic(joinpath(root, "XOR", "two-input-2x2-hidden-majority-vote-baseline")),
        save_xor_checkerboard_schematic(joinpath(root, "XOR", "checkerboard-local-cnn-two-hidden-layers")),
        save_xor_edge_schematic(joinpath(root, "XOR", "edge-driven-single-layer-readout")),
        save_mnist_local_schematic(joinpath(root, "MNIST", "single-hidden-local-28x28-to-11x11-readout")),
        save_mnist_cnn_schematic(joinpath(root, "MNIST", "two-hidden-local-28x28-to-14x14-readout")),
        save_mnist_inlaid_schematic(joinpath(root, "MNIST", "inlaid-55x55-pixel-readout")),
    ])
    for path in outputs
        println(path)
    end
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
