# Map special symbols on where to find them in the args

function refmap(::Val{:w})
    return (:gadj,)
end

function refmap(::Val{:self})
    return (:g, :self)
end

function refmap(::Val{:s})
    return (:gstate,)
end

function refmap(::Val{:sn})
    return (:newstate,)
end

function refmap(::Val{A}) where A
    return (:hamiltonian,:($A))
end