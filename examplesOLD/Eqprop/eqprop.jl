using InteractiveIsing

g = IsingGraph(
    Layer(200,200),
    @WG dr -> 
    Layer(10,10),
    Layer(5,5)
)

createProcess(g)
interface(g)
