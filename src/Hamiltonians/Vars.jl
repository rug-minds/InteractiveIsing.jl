
"""
Add factors to factor array defined in module scope
"""
macro addfactor(facnames...)
    for name in facnames
        push!(factors,eval(name))
    end
end

unweightedloop = hFactor("-gstate[connIdx(conn)]" , :Weighted, false, :FacLoop)
weightedloop = hFactor("-connW(conn)*gstate[connIdx(conn)]", :Weighted, true, :FacLoop)
magfac = hFactor("-mlist(g)[idx]", :MagField, true, :FacTerm)
clampfac = hFactor("clampparam(g)/2 * (newstate^2-oldstate^2 - 2 * @inbounds clamps(g)[idx] * (statediff))", :Clamp, true, :DiffTerm)
defects = hFactor("", :Defects, false, :Etc)

factors = [];
export factors

@addfactor unweightedloop weightedloop magfac clampfac defects 

terms = []