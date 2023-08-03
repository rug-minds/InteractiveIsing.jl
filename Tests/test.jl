struct Inner
    somearray
    somefloat
end

struct Outer
    someint
    istruct
end

struct Inner2
    somearray
    somefloat
end

struct Outer2
    someint::Int64
    istruct
end

struct Inner3
    somearray::Array{Float64}
    somefloat::Float64
end

struct Outer3
    someint
    istruct::Inner3
end

struct Inner4
    somearray::Array{Float64}
    somefloat::Float64
end

struct Outer4
    someint::Int64
    istruct
end

struct Inner5
    somearray
    somefloat
end

struct Outer5
    someint::Int64
    istruct::Inner5
end

struct Inner6
    somearray::Array{Float64}
    somefloat::Float64
end

struct Outer6
    someint::Int64
    istruct::Inner6
end

function test1(out)
    in = out.istruct
    ti = time()
    for _ in 1:10000000
        algo(out.someint, in)
    end
    tf = time()
    println(tf - ti)
end

function test2(out , in)
    ti = time()
    for _ in 1:10000000
        algo(out.someint, in)
    end
    tf = time()
    println(tf - ti)
end

function test3(out)
    someint = out.someint
    in = out.istruct
    ti = time()

    function algo()
        in.somearray[someint] = someint*in.somefloat
    end

    for _ in 1:10000000
        algo()
    end
    tf = time()
    println(tf - ti)
end

function test14(out)
    in::Inner4 = out.istruct
    ti = time()
    for _ in 1:10000000
        algo(out.someint, in)
    end
    tf = time()
    println(tf - ti)
end

function test34(out)
    someint = out.someint
    in::Inner4 = out.istruct
    ti = time()

    function algo()
        in.somearray[someint] = someint*in.somefloat
    end

    for _ in 1:10000000
        algo()
    end
    tf = time()
    println(tf - ti)
end

function test4(out)
    test2(out, out.istruct)
end

function algo(someint , in)
    in.somearray[someint] = someint*in.somefloat
end



const inner = Inner([1.,2.,3.], 1.5)
const outer = Outer(1,inner)

const inner2 = Inner2([1.,2.,3.], 1.5)
const outer2 = Outer2(1,inner2)

const inner3 = Inner3([1.,2.,3.], 1.5)
const outer3 = Outer3(1,inner3)

const inner4 = Inner4([1.,2.,3.], 1.5)
const outer4 = Outer4(1,inner4)

const inner5 = Inner5([1.,2.,3.], 1.5)
const outer5 = Outer5(1,inner5)

const inner6 = Inner6([1.,2.,3.], 1.5)
const outer6 = Outer6(1,inner6)


# Tests
function runtests()
    println("Test 1")
    test1(outer)
    test2(outer,outer.istruct)
    test3(outer)
    test4(outer)
    println("Test 2")
    test1(outer2)
    test2(outer2,outer2.istruct)
    test3(outer2)
    test4(outer2)
    println("Test 3")
    test1(outer3)
    test2(outer3,outer3.istruct)
    test3(outer3)
    test4(outer3)
    println("Test 4")
    test1(outer4)
    test14(outer4)
    test2(outer4,outer4.istruct)
    test3(outer4)
    test34(outer4)
    test4(outer4)
    println("Test 5")
    test1(outer5)
    test2(outer5,outer5.istruct)
    test3(outer5)
    test4(outer5)
    println("Test 6")
    test1(outer6)
    test2(outer6,outer6.istruct)
    test3(outer6)
    test4(outer6)
end

runtests()