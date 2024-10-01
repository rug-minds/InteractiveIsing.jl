waittime = 1

function createtask()
    ti = time()
    Threads.@spawn while time() - ti < waittime
        sleep(1)
    end
end

t = createtask()
w = @async wait(t)

function waittimed()
end

