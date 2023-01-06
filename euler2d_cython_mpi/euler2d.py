#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check for MPI
Has_MPI=True
try:
    import mpi4py.MPI as MPI
except ImportError:
    Has_MPI=False


import configparser
import sys

import numpy as np
import matplotlib.pyplot as plt

from euler2d import hydroMonitoring
from euler2d import hydroParam as hp
from euler2d import hydroRun


# ################################
# EULER2D MAIN
# ################################
if __name__ == '__main__':

    # MPI related vars.
    comm = None
    nProc = 1
    myRank = 0
    if Has_MPI:
        comm = MPI.COMM_WORLD
        nProc = comm.Get_size()
        myRank = comm.Get_rank()

    if len(sys.argv) > 1:
        paramFilename = sys.argv[1]
    else:
        paramFilename = './test_implode.ini'

    # parser input parameter file
    Config = configparser.ConfigParser()
    Config.read(paramFilename)
    if myRank == 0:
        print(Config)

    t  = 0.0
    dt = 0.0
    nStep = 0

    # read parameter file
    par = hp.hydroParams(paramFilename)
    if myRank==0:
        par.printConfig()

    # sanity check : MPI parameters valid ?
    if par.mx * par.my != nProc:
        print("Error: total number of MPI proc ({}) doesn't match product mx tes my={} !!!".format(nProc,par.mx * par.my))


    # create the main data structure : a hydroRun (a changer --> dico)
    parDict = par.getDict()
    hr = hydroRun.hydroRun(parDict)

    # initial condition
    hr.init_condition()

    # initialize time step
    dt = hr.compute_dt(0)
    print("dt = {}".format(dt))

    # initialize boundaries
    hr.make_boundaries_internal(0)
    hr.make_boundaries_internal(1)
    hr.make_boundaries_external(0)
    hr.make_boundaries_external(1)

    # start computation
    if myRank==0:
        print("Start computation....")
    hydroMonitoring.total_timer.start()

    # Hydrodynamics solver loop
    t     = 0.0
    nStep = 0
    while t < parDict['tEnd'] and nStep < parDict['nStepmax']:
        # output
        hydroMonitoring.io_timer.start()
        if nStep % parDict['nOutput'] == 0:
            if myRank==0:
                print("Output results at time t={0:16.13f} step {1:05d} dt={2:13.10f}".format(t,nStep,dt))
            filename = "euler2d_mpi_{0:07d}_step_{1:07d}.vti".format(myRank,nStep)
            if nStep % 2 == 0:
                hr.saveVTK(hr.U,  bytes(filename, encoding='utf-8'), nStep)
            else:
                hr.saveVTK(hr.U2, bytes(filename, encoding='utf-8'), nStep)

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
    t_dt    = hydroMonitoring.dt_timer.elapsed()
    t_bound = hydroMonitoring.boundaries_timer.elapsed()
    t_io    = hydroMonitoring.io_timer.elapsed()

    # MPI reduce
    t_tot_global   = comm.reduce(sendobj=t_tot, op=MPI.MAX)
    t_comp_global  = comm.reduce(sendobj=t_comp, op=MPI.MAX)
    t_dt_global    = comm.reduce(sendobj=t_dt, op=MPI.MAX)
    t_bound_global = comm.reduce(sendobj=t_bound, op=MPI.MAX)
    t_io_global    = comm.reduce(sendobj=t_io, op=MPI.MAX)

    if myRank==0:
        print("total         time : {0:5.3f} secondes".format(t_tot_global))
        print("compute       time : {0:5.3f} secondes {1:6.3f}%".format(t_comp_global,t_comp_global/t_tot_global*100))
        print("  dt          time : {0:5.3f} secondes {1:6.3f}%".format(t_dt_global,t_dt_global/t_tot_global*100))
        print("  boundaries  time : {0:5.3f} secondes {1:6.3f}%".format(t_bound_global,t_bound_global/t_tot_global*100))
        print("  io          time : {0:5.3f} secondes {1:6.3f}%".format(t_io_global,t_io_global/t_tot_global*100))
        print("Perf               : {0:10.2f} number of cell-updates/s".format(nStep*par.nx*par.mx*par.ny*par.my/t_tot_global))
