function getindex(l::IsingLayer{A,B,C,D,T}, idx) where {A,B,C,D,T}
    coords_3d = coords(l)

    if !isnothing(coords_3d)
        coords_3d = (0,0,0)
    end

    len_2d = size(l,1)
    width_2d = size(l,2)
    nstates_2d = len_2d*width_2d
    start = startidx(l) + nstates_2d*(idx-1)
    return IsingLayer(statetype(l), 
                        graph(l), 
                        internal_idx(l),
                        start,
                        len_2d,
                        width_2d,
                        lsize = size(l)[1:2],
                        set = stateset(l),
                        name =  "2d Layer $idx of 3d layer $(name(l))", 
                        coords = Coords(y = coords_3d[1], x = coords_3d[2], z = coords_3d[3] + idx -1))
end

Base.eachindex(l::AbstractIsingLayer{T,3}) where T = Base.OneTo(size(l,3))
