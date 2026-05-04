using InteractiveIsing
using InteractiveIsing.Windows

function connection_weights_2d(; dr)
    return dr == 0 ? 0.0f0 : Float32(exp(-0.55 * dr))
end

function connection_weights_3d(; dr)
    return dr == 0 ? 0.0f0 : Float32(exp(-0.45 * dr))
end

wg2d = @WG connection_weights_2d NN = 4
wg3d = @WG connection_weights_3d NN = (2, 2, 1)

g2d = IsingGraph(
    Layer(56, 36, Continuous(), wg2d);
    precision = Float32,
)

host2d = window(title = "2D Selected Adjacency Connections", size = (1100, 900))
panel!(
    host2d,
    ConnectionsPanel(
        g2d;
        selected_nodes = [(1, (8, 8)), (1, (28, 18)), (1, (48, 30))],
        selection_mode = :incident,
        max_edges = 5_000,
        curved = true,
        curve_amount = 0.08,
        curve_resolution = 7,
        colormap = :viridis,
        line_kwargs = (; linewidth = 1),
        node_kwargs = (; color = (:dodgerblue, 0.55), markersize = 3),
    ),
    (1, 1),
)

g3d = IsingGraph(
    24, 16, 8,
    Continuous(),
    wg3d,
    LatticeConstants(1.0f0, 1.0f0, 1.0f0),
    StateSet(-1.0f0, 1.0f0);
    precision = Float32,
)

host3d = window(title = "3D Selected Adjacency Connections", size = (1100, 900))
panel!(
    host3d,
    ConnectionsPanel(
        g3d;
        selected_nodes = [(1, (1, 1, 1)), (1, (12, 8, 4)), (1, (20, 12, 7))],
        selection_mode = :incident,
        max_edges = 7_500,
        curved = true,
        curve_amount = 0.12,
        curve_resolution = 6,
        colormap = :plasma,
        line_kwargs = (; linewidth = 1),
        node_kwargs = (; color = (:darkorange, 0.5), markersize = 2.5),
    ),
    (1, 1),
)
