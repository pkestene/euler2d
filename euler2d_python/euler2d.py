#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys

import numpy as np
import matplotlib.pyplot as plt

from euler2d import hydroMonitoring

# #########################
# MAIN
# #########################
def main(paramFilename):

    # parser input parameter file
    Config = ConfigParser.ConfigParser()
    Config.read(paramFilename)
    #print Config.sections()
    print Config

    t  = 0.0
    dt = 0.0
    nStep = 0

    from euler2d import hydroParam as hp
    par = hp.hydroParams(paramFilename)

    par.printConfig()


    from euler2d import hydroRun
    
    # create the main data structure : a hydroRun
    hr = hydroRun.hydroRun(paramFilename)

    # initial condition
    hr.init_condition()

    # initialize time step
    dt = hr.compute_dt(0)
    #print "dt = {}".format(dt)

    # initialize boundaries
    hr.make_boundaries(0)
    hr.make_boundaries(1)

    # start computation
    print "Start computation...."
    hydroMonitoring.total_timer.start()

    # Hydrodynamics solver loop
    t     = 0.0
    nStep = 0
    while t < hr.param.tEnd and nStep < hr.param.nStepmax:
        # output
        hydroMonitoring.io_timer.start()
        if nStep % hr.param.nOutput == 0:
            print "Output results at time t={0:16.13f} step {1:05d} dt={2:13.10f}".format(t,nStep,dt)
            filename = "U_{0:03d}".format(nStep)
            if nStep % 2 == 0:
                hydroRun.saveVTK(hr.U,  filename)
            else:
                hydroRun.saveVTK(hr.U2, filename)

        hydroMonitoring.io_timer.stop()

        # compute new dt
        dt =  hr.compute_dt(nStep%2)
    
        # perform one step integration
        hr.godunov_unsplit(nStep, dt)

        # increase time
        nStep += 1
        t+=dt
    
    
    hydroMonitoring.total_timer.stop()

    # print monitoring information
    t_tot   = hydroMonitoring.total_timer.elapsed()
    t_comp  = hydroMonitoring.godunov_timer.elapsed()
    t_bound = hydroMonitoring.boundaries_timer.elapsed()
    t_io    = hydroMonitoring.io_timer.elapsed()
    print "total       time : {0:5.3f} secondes".format(t_tot)
    print "compute     time : {0:5.3f} secondes {1:6.3f}%".format(t_comp,t_comp/t_tot*100.0)
    print "boundaries  time : {0:5.3f} secondes {1:6.3f}%".format(t_bound,t_bound/t_tot*100.0)
    print "io          time : {0:5.3f} secondes {1:6.3f}%".format(t_io,t_io/t_tot*100.0)
    print "Perf             : {0:10.2f} number of cell-updates/s".format(nStep*par.isize*par.jsize/t_tot)
  

# #########################
# MAIN WITH CALLGRAPH
# #########################
def main_with_callgraph(paramFilename):

    from pycallgraph import PyCallGraph
    from pycallgraph import Config
    from pycallgraph import GlobbingFilter
    from pycallgraph.output import GraphvizOutput

    config = Config()
    config.trace_filter = GlobbingFilter(exclude=[
        'vtk.*',
        'evtk.*',
        'imageToVTK',
    ])
    graphviz = GraphvizOutput()
    graphviz.output_file = 'euler2d_callgraph.png'

    with PyCallGraph(output=graphviz, config=config):
        main()

# ################################
# EULER2D MAIN
# ################################
if __name__ == '__main__':

    if len(sys.argv) > 1:
        paramFilename = sys.argv[1]
    else:
        paramFilename = './test_implode.ini'

    print('Using param {}'.format(paramFilename))
        
    main(paramFilename)
    #main_with_callgraph(paramFilename)
