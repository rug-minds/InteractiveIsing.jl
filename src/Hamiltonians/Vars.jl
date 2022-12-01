
"""
Add factors to factor array defined in module scope
"""
macro addfactor(facnames...)
    for name in facnames
        push!(factors,eval(name))
    end
end

unweightedloop = hFactor("-g.state[connIdx(conn)]" , :Weighted, false, :FacLoop)
weightedloop = hFactor("-connW(conn)*g.state[connIdx(conn)]", :Weighted, true, :FacLoop)
magfac = hFactor("-g.d.mlist[idx]", :MagField, true, :FacTerm)
clampfac = hFactor("beta(g)/2 * (newstate^2-oldstate^2 - 2 * clamps(g)[idx](statediff) )", :Clamp, true, :DifTerm)
defects = hFactor("", :Defects, false, :Etc)

factors = [];

@addfactor unweightedloop weightedloop magfac clampfac defects 

terms = []