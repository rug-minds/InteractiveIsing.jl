"""
Save magnetization data to a dataframe
"""
function MToDF(M, dpoint::Integer = Int32(1), T::Real = Float16(1))
    return DataFrame(M = M, D = dpoint, T = T)
end

""" 
Determine amount of datapoints per temperature from dataframe data (if homogeneous over temps)
"""
function detDPoints(dat)
    dpoint = 0
    lastt = dat[:,1][1]
    for temp in dat[:,1]
        if temp == lastt
            dpoint += 1
        else 
            return dpoint
        end
    end
    return dpoint
end

"""
Input the temperature sweep data into a dataframe
"""
function dataToDF(tsData, lMax = length(tsData[2][1]))
    return DataFrame(Temperature = tsData[1], Correlation_Function = [corrLXY[1:lMax] for corrLXY in  tsData[2]], Magnetization = tsData[3] )
end

"""
Read CSV and outputs dataframe
Make this use julia data format
"""
csvToDF(filename) = DataFrame(CSV.File(filename)) 

"""
Helper Functions 
"""
function insertShift(vec::Vector{T}, el) where T
    newVec = Vector{T}(undef, length(vec))
    newVec[1:(end-1)] = vec[2:end]
    newVec[end] = T(el)
    return newVec
end

""" 
Create folder in data folder with given name and return path
"""
function dataFolderNow(dataString::String)
    nowtime = "$(now())"[1:(end-7)]
    try; mkdir(joinpath(dirname(Base.source_path()),"Data")); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), "Data", dataString)); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), "Data", dataString, "$nowtime")); catch end
    return "Data/Tempsweep/$nowtime/"
end

"""
Try to make a folder and don't do anything if it's already there
"""
function trymakefolder(foldername::String)
    if foldername[end] == '/'
        foldername = foldername[1:(end-1)]
    end
    try; mkdir(joinpath(dirname(Base.source_path()),foldername)); catch end
end