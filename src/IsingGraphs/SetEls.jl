"""
Adding a defect to lattice
"""
# Decide wether to set or clamp and set the sim type
function setOrClamp!(sim, g::AbstractIsingGraph{T}, idxs , brush, clamp = false) where T

    if T == Int8
        clamp = brush == 0 ? true : clamp
    end
    # Set the defects
    setrange!(defects(g), clamp, idxs)
    
    # Set the htype to wether it has defects or not
    setSimHType!(sim, :Defects => hasDefects(graph(g)))
    
    # Set the spins
    @inbounds state(g)[idxs] .= brush
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




    