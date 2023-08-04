using GLMakie

struct Container
    m::Matrix{Float32}
end

const c = Container(rand(500,500))
const obc = Observable(c.m)

on(obc) do m
    println("Observable changed")
end


f = Figure()
ax = Axis(f[1,1])
scene = image!(ax, obc, colormap = :thermal)
GLMakie.activate!(inline=false)

display(f)

c.m .= rand(500,500)
notify(obc)