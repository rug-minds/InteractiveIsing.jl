#TODO: Create a cleaner interface for plotting

"""
Plot and save correlation data from a vec with sampled lengths and vec with correlation data for those lengths
the other arguments are for the file names 
"""
function plotCorr(lVec, corrVec; save = true, dodisplay = true, foldername = "Data", lidx = 0, tidx = 0, point = 0, T = 0)
    corrPlot = pl.plot(lVec,corrVec, xlabel = "Length", label = L"C(L)")
    
    if dodisplay
        display(corrPlot)
    end

    if save
        Tstring = replace("$T", '.' => ',')
        pl.savefig(corrPlot,"$(foldername)Ising Corrplot L$lidx $tidx T$Tstring d$point")
    end
end

plotCorr(vecs::Tuple; kwargs...) = let (lVec,corrVec) = vecs; plotCorr(lVec,corrVec; kwargs...) end

plotCorr(layer::IsingLayer; save = false, kwargs...) = plotCorr(correlationLength(layer); save, kwargs...)
export plotCorr