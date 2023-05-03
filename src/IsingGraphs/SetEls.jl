"""
Adding a defect to lattice
"""
# Decide wether to set or clamp and set the sim type
function setOrClamp!(g::AbstractIsingGraph{T}, idxs , brush, clamp = false) where T

   
end

"""
Removes all defects from graph and branch sim
"""
function restoreState!(sim, g)
    (@view state(g)[defectList(g)]) .= rand(length(defectlist(g)))
    reset!(defects(g))
    editHType!(g, :Defects => false)
    branchSim(sim)
end




    