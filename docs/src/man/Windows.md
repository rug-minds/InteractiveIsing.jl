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

which can be listened to using the usual syntax from [Observables.jl](https://juliagizmos.github.io/Observables.jl/stable/)


## Lines Window

The lines window is a window type that create a figure based on the `lines` function in Makie. Since plotting static data using makie is simplest done through Makie itself, the lines window instead handles interactive plots where data is gathered on a separate thread and continuously updated.

To do this, the lines window wants a process, specifically one that updates two vectors called `x` and `y`. This kind of process can be obtained from the function

```
wp = linesprocess(step_function, number_of_repeats)
```
The first agrument it wants is a function that accepts a named tuple as argument, which will hold the following data: 

* `process`: The process itself
* `x`   : A vector holding the x coordinate data
* `y`   : A vector holding the y coordinate data

The function itself desribes how the `x` and `y` data are updated every step. The index of the loop in which the process currently is can be obtained from `loopidx(process)`. To get access to the data, use tuple unpacking like follows

```
function step_function(args)
    (;process, x, y) = args

    # Mutate x and y

end
```

The `linesproces` function may be wrapped into another function if data needs to be prepared

```
function make_some_linesprocess(arg1,arg2, ...)
    # Prepare data
    some_data = getdata(arg1, arg2)
    return linesprocess(repeats) do args
        (;process, x, y) = args

        # Some data
        do_something!(some_data)
    end
end
```

This way the lines process will have a local copy to the data `some_data`.

