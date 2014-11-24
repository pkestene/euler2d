# -*- coding: utf-8 -*-

import time


class hydroTimer(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = 0.0
        self.total_time = 0.0
        self.running = 0
        
    def start(self):
        self.start_time = time.time()
        self.running = 1

    def stop(self):
        """
        Stop timer doesn't do anything if it has already been stopped.
        """
        if self.running:
            stop_time        = time.time()
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
