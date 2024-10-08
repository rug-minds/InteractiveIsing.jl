# Windows

Windows are an abstraction on top of GLMakie's screen system. A window is an object that keeps track of timers, processes and Makie objects. If it is closed through "WIN+W" (Windows) or "CMD+W" (Mac) then it will take care of lingering timers, processes and data in the background.

We will provide multiple distinct functions that create windows for specific functions.

The window struct looks as follows
```
mutable struct MakieWindow <: AbstractWindow
    uuid::UUID
    f::Figure
    screen::GLMakie.Screen
    timers::Vector{PTimer}
    other::Dict{Symbol, Any}
end
```

Windows may be indexed directly as a dict using symbols as keys. The index is forwarded to the field `other`.

```
w = makesomewindow()
w[:data] #Gets some data
```

The data that is stored in the window will depend on how the window is constructed, i.e. which constructor function is used.

A window always holds an observable that tracks wether the window is open

```
wo = window_open(w)

on(wo) do
    # Do something when window closes
end
```

which can be listened to using the usual syntax from [Observables.jl](@extref https://juliagizmos.github.io/Observables.jl/stable/)


## Lines Window

