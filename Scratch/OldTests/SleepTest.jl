const vec = rand(10000)
const shouldrun = Ref(true)

uselessalgo(vec) = vec .= rand(10000)

function uselessLoop(vec, shouldrun)
    while shouldrun[]
        uselessalgo(vec)
        GC.safepoint()
    end
end

function spawnUselessLoop(vec, shouldrun)
    # println("Starting loop...")
    shouldrun[] = true
    # println("Spawning thread...")
    Threads.@spawn uselessLoop(vec, shouldrun)
    # println("Sleeping")
    sleep(2)
    # println("Stopping loop...")
    shouldrun[] = false
end

# spawnUselessLoop(vec, shouldrun)