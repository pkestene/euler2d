import hydroMonitoring as hydroM


def test():
    t=hydroM.hydroTimer()

    t.start()

    t.elapsed()
    print("quelque chose")

    ii=0
    for i in xrange(1,10000):
        ii = ii + i
    t.stop()

    t1=t.elapsed()
    print("elapsed 1: {}".format(t1))

    ii=0
    for i in xrange(1,10000):
        ii = ii + i
    t.stop()

    t2=t.elapsed()
    print("elapsed 2: {}".format(t2))
    assert(t1==t2)
