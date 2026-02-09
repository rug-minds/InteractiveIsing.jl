shouldrun = Ref(true)
function SimpleMainLoop(process, g, state, adj)
    while shouldrun()
        updatemontecarlo!(g, state, adj)
    end
        

end