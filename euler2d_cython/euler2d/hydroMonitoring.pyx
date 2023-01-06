# -*- coding: utf-8 -*-

# cython: language_level = 3

#from posix.time  cimport timeval


# if posix.time unavailable
cdef extern from "sys/types.h":

    ctypedef long time_t
    ctypedef long suseconds_t

cdef extern from "sys/time.h" nogil:

    cdef struct timezone:
        int tz_minuteswest
        int dsttime

    cdef struct timeval:
        time_t      tv_sec
        suseconds_t tv_usec

    int gettimeofday(timeval *tp, timezone *tzp)

########################################################
# `hydroTimer` class
########################################################
cdef class hydroTimer:
    """
    A timer class based on C routine gettimeofday.
    """

    cdef public double start_time, total_time
    cdef public int running

    def __init__(self):

        self.reset()

    def reset(self):

        self.start_time = 0.0
        self.total_time = 0.0
        self.running = 0

    def start(self):
        cdef timeval now
        gettimeofday(&now, NULL)
        self.start_time = 1.0*now.tv_sec+(1e-6)*now.tv_usec
        self.running = 1

    def stop(self):
        cdef double stop_time
        cdef timeval now

        if self.running:

            gettimeofday(&now, NULL)
            stop_time = 1.0*now.tv_sec+(1e-6)*now.tv_usec
            self.total_time += (stop_time - self.start_time)
            self.running=0

    def elapsed(self):
        """
        Return total accumulated time in seconds.
        """
        return self.total_time


# define global timers
total_timer      = hydroTimer()
godunov_timer    = hydroTimer()
boundaries_timer = hydroTimer()
io_timer         = hydroTimer()
test_timer       = hydroTimer()
