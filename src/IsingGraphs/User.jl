# Setting elements

"""
Set spins either to a value or clamp them
"""

setSpins!(sim, g, idxs, brush, clamp = false) = setGraphSpins!(sim, g, idxs, brush, clamp)
setSpins!(sim, g, tupls, brush, clamp = false) = setGraphSpins!(sim, g, coordToIdx.(tupls, glength(g)), brush, clamp)


