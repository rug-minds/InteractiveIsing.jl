using GLMakie
using Observables

f = Figure()
display(f)

const state1 = rand(28,28)
const state2 = rand(32,32)
const state3 = rand(16,49)

const imgob = Observable(state1)

ax = Axis(f[1,1])
imgref = Ref(image!(ax, imgob))

# t = Timer(1/60, interval = 1/60) do timer
#     notify(imgob)
# end

shouldrun = Ref(true)
# @async while shouldrun[]
#     state1 .= rand(28,28)
#     state2 .= rand(32,32)
#     state3 .= rand(16,49)
#     yield()
# end

function set_newstate(state)
    delete!(ax,imgref[])
    imgob[] = state
    imgref[] = image!(ax, imgob)
    reset_limits!(ax)
end
