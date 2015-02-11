# for calling gettimeofday in libc
type Timeval_t
    tv_sec::Clong
    tv_usec::Cuint
end

# our timer type
type Timer_t
    start_time::Timeval_t
    total_time::Float64

    # usefull default constructor
    Timer_t() = new(Timeval_t(0,0), 0.0)
end

# wrap around gettimeofday in C
function get_time_of_day(timeval::Timeval_t)
    #timeval = Array(Timeval_t, 1)
    ccall(:gettimeofday, Cint, (Ptr{Void}, Ptr{Void}), &timeval, C_NULL)
    return timeval
end


# reset timer
function timer_reset(t::Timer_t)

    timeval = t.timevalart_time
    timeval.tv_sec = 0
    timeval.tv_usec = 0

end

# start timer
function timer_start(t::Timer_t)

    timeval = t.start_time
    typeof(timeval)
    get_time_of_day(timeval)

    #println(t.total_time)
end

# stop timer
function timer_stop(t::Timer_t)

    accum = 0.0
    start = t.start_time
    now = Timeval_t(0,0)

    get_time_of_day(now)
    
    # accumulate time between last start and now
    if (now.tv_sec == start.tv_sec)
        accum = Float64(now.tv_usec - start.tv_usec) * 1e-6
    else
        accum = Float64(now.tv_sec - start.tv_sec) + Float64(now.tv_usec - start.tv_usec) * 1e-6
    end

    t.total_time += 1.0*accum

    #println(t.total_time)

end

# elapsed time
function elapsed(t::Timer_t)

    return t.total_time

end
