function newmakie(makietype, args...; kwargs...)
    f = makietype(args...; kwargs...)
    scr = GLMakie.Screen()
    display(scr, f)
    return f
end

function makieaxis(axisfunc, modifiers...)
    f = Figure()
    ax = axisfunc(f[1, 1])
    for mod in modifiers
        mod(ax)
    end
    scr = GLMakie.Screen()
    display(scr, f)
    return f
end
