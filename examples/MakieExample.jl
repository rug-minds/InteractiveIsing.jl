using GLMakie





const m = rand(500,500)
const obm = Observable(m)

f = Figure()
ax = Axis(f[1,1])
scene = image!(ax, obm, colormap = :thermal)
GLMakie.activate!(inline=false)
display(f)

tm = Timer((timer) -> notify(obm), 0., interval = 1/60)

Threads.@spawn while true
    m .= rand(500,500)
    GC.safepoint()
end