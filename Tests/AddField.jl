using Observables
mutable struct Test
    a
end

const pdict = Dict{Ptr{Nothing}, Observable{Any}}()
t = Test(1)
function addfield(t::Test, val)
    pdict[pointer_from_objref(t)] = Observable(val)

    return nothing
end

gf(t::Test) = pdict[pointer_from_objref(t)]


addfield(t, 2)

gf(t)