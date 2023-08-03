using GLMakie, Observables

struct Container
    m::Matrix{Float32}
end

function update(obs)
    obs[] = obs[]
    return
end

const m = rand(500,500)
const cm = Container(m)
const c = Container(rand(500,500))

const obm = Observable(m)
const obcm = Observable(cm.m)
const obc = Observable(c.m)

f = Figure()
ax = Axis(f[1,1])
scene = heatmap!(ax, obc, colormap = :thermal)



display(f)

tm = Timer((timer) -> update(obc) ,0., interval = 1/60)

function spawnloop(mat::Matrix)
    Threads.@spawn while true
        mat .= rand(500,500)
        GC.safepoint()
    end
end

function spawnloop(con::Container)
    Threads.@spawn while true
        con.m .= rand(500,500)
        GC.safepoint()
    end
end

spawnloop(c)