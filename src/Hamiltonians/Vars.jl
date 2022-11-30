unweightedloop = hFactor("-g.state[connIdx(conn)]" , :Weighted, false, true)
weightedloop = hFactor("-connW(conn)*g.state[connIdx(conn)]", :Weighted, true, true)
magfac = hFactor("-g.d.mlist[idx]", :MagField, true, false)
clampfac = hFactor("g.d.clampfac[idx]*g.state[idx]", :Clamp, true, false)
defects = hFactor("", :Defects, false, false)

factors = [];

@addfactor unweightedloop weightedloop magfac clampfac defects 

terms = []