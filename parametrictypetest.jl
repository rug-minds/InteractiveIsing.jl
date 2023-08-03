struct Inner{A,B} end
struct Outer 
    i::Inner
end


t1 = Outer(Inner{true,false}())
t2 = Outer(Inner{true,true}())

function func1(outer, inner = outer.i)
    println(typeof(inner).parameters)
end

function func1(outer, inner::Inner{A,true} = outer.i) where A
    println("true")
    func1(outer)
end

func1(t1)
func1(t2)

