# Interactive Simulations

Interactive simulations let the window control selected process variables while
the simulation is running. The graph stores the requested controls, and
`createProcess(g)` turns them into `Processes.Interactive` context variables for
the running algorithm.

The basic pattern is:

```julia
using InteractiveIsing

g = IsingGraph(48, 48, Continuous())
g.default_algorithm = LocalLangevin(
    stepsize = 0.05f0,
    max_drift_fraction = 0.2f0,
    adjusted = true,
)

g.addons[:interactive] = true
temp!(g, 0.15f0)

interactivevar!(
    g,
    LocalLangevin,
    :stepsize;
    value = 0.05f0,
    range = 0.0:0.0025:0.25,
    label = "stepsize",
)

host = interface(g; framerate = 30, polling_rate = 10)
process = createProcess(g)
```

`g.addons[:interactive] = true` enables the standard interactive temperature
path. When the algorithm has a temperature variable (`:T` or `:temp`), that
variable is marked interactive in the process context. The temperature panel
then writes to the graph temperature, and the running dynamics reads the updated
interactive variable through the process context.

Use `interactivevar!` for additional algorithm variables, such as Langevin
`stepsize` or `max_drift_fraction`. These variables are stored in
`g.addons[:interactive_vars]`, so they are part of the graph configuration and
are applied whenever a new process is created.

## Registering Variables

```julia
interactivevar!(
    g,
    target,
    varname;
    value = nothing,
    range = nothing,
    label = string(varname),
)
```

The arguments mean:

- `target` is the algorithm target whose process context owns the variable. For
  a simple graph process, using the algorithm type is usually enough:
  `LocalLangevin`, `Metropolis`, or `KineticMC`.
- `varname` is the context variable that should become interactive.
- `value` is an optional initial value. If present, `createProcess(g)` inserts a
  matching `Processes.Override` before marking the variable interactive.
- `range` is the UI range. Prefer an explicit range for user-facing controls,
  because it gives the slider stable bounds and step size.
- `label` is the text shown in the interactive variables panel.

The target should be the dynamics that owns the variable. For example,
Langevin-specific controls should target `LocalLangevin`, not the graph or a
display panel.

When the process is built, the target is resolved against the prepared algorithm
registry. A broad type target works when it identifies one concrete algorithm in
the current process. If a composite process contains multiple algorithms of the
same type, use a more specific target so the variable is not ambiguous.

Existing registrations can be inspected with:

```julia
interactivevars(g)
```

Calling `interactivevar!` again with the same `target` and `varname` replaces
the existing registration.

## UI Controls

`interface(g)` mounts optional simulation panels based on the graph and the
process shape.

The standard temperature panel is shown in the simulation interface. If
`g.addons[:interactive] = true`, process creation marks the algorithm
temperature as interactive, so the running dynamics sees UI changes.

The interactive variables panel is shown only when `interactivevars(g)` is not
empty. It creates one row per registered variable, except for temperature names
handled by the temperature panel. Each row contains:

- a slider for the variable value;
- a delta textbox;
- `-` and `+` buttons that subtract or add the current delta;
- hold behavior on the buttons, so keeping a button pressed repeats the step.

The delta textbox is committed when a step button is clicked, so pressing Enter
is not required before using `-` or `+`.

If a `range` is supplied, the slider uses that range and the step buttons snap
to its grid. This avoids asymmetric up/down changes caused by floating point
values drifting away from the slider step. If no range is supplied, the panel
uses a small heuristic around the current value, but explicit ranges are better
for examples and stable interfaces.

## Reset

The reset button in the status panel resets the graph state. If the graph had a
running process, reset closes it, resets the graph, and starts a fresh process
with the graph's current process configuration.

This matters for interactive simulations because the graph-level registrations
and stored interactive values remain on `g.addons`. After reset, the new process
is created with the same interactive variables and overrides.

## Conditional Panels

Optional panels are loaded through the simulation panel support predicates. The
generic hook is:

```julia
InteractiveIsing.Windows._panel_supported(g, Val(:panel_key))
```

and panels are mounted with:

```julia
InteractiveIsing.Windows._mount_panel_if_supported!(
    handle,
    :panel_key,
    () -> MyPanel(g),
    cell,
)
```

The default predicate returns `true`, so ordinary panels keep their existing
behavior. Specialized optional panels can define a narrower predicate.

Current conditional panels include:

- `:interactive_variables`, mounted when `interactivevars(g)` is not empty;
- `:kinetic_time`, mounted when the graph has a live kinetic-time snapshot or
  the active/default algorithm tree contains `KineticMC`.

The predicate inspects the latest running graph process when one exists.
Otherwise it falls back to `g.default_algorithm`, which lets the UI build the
right panel layout before `createProcess(g)` is called.

## Langevin Example

The repository contains a runnable example:

```julia
include("examples/Interactive Langevin.jl")
```

It opens a continuous-spin Ising simulation with:

- interactive temperature through `g.addons[:interactive] = true`;
- an interactive `LocalLangevin` step-size control;
- an interactive `LocalLangevin` drift-cap control;
- explicit slider ranges and delta step controls for the Langevin variables.

The example function returns `nothing`, so including the file does not dump the
graph, host, and process internals into the REPL.
