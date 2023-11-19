using GLMakie, SparseArrays
function creategrid(conns, imsize, gridsize = (10,10))
    f, screen, isopen = empty_window()

    println("Imgsize = $imsize")
    grid = GridLayout(f[1:gridsize[1],1:gridsize[2]])
    for i in 1:gridsize[1]
        for j in 1:gridsize[2]
            mat = reshape(Vector(conns[:,i*(gridsize[1]-1)+mod1(j,gridsize[2])]), imsize)
            image(grid[i,j], mat, colormap = :thermal)
        end
    end
end