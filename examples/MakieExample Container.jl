using GLMakie

struct Container
    m::Matrix{Float32}
end

const c = Container(rand(500,500))
const obc = Observable(c.m)

f = Figure()
ax = Axis(f[1,1])
scene = image!(ax, obc, colormap = :thermal)
GLMakie.activate!(inline=false)

display(f)

tm = Timer((timer) -> notify(obc), 0., interval = 1/60)

Threads.@spawn while true
    c.m .= rand(500,500)
    GC.safepoint()
end