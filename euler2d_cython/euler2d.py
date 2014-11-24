#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import sys

import numpy as np
import matplotlib.pyplot as plt

from euler2d import hydroMonitoring
from euler2d import hydroParam as hp
from euler2d import hydroRun

# test if vtk module is available
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    vtkModuleFound = True
except ImportError:
    vtkModuleFound = False

# test if pyEVTK is available
# See https://bitbucket.org/pauloh/pyevtk
try:
    import evtk
    evtkModuleFound = True
except:
    evtkModuleFound = False

    
def saveVTK(U, filename):
    """
    Main IO routine. Save U array into a vtk file.
    filename should include .vti suffix
    """

    # enum for hydro variables (taken from hydroParam)
    ID = hp.ID
    IP = hp.IP
    IU = hp.IU
    IV = hp.IV

    # filename without suffix (.vti)

    # use vtk module if available
    if vtkModuleFound:
        #writer = vtk.vtkXMLImageDataWriter()
        #writer.SetFileName(filename)
        
        # TO BE CONTINUED
        pass

    # use evtk module if available
    if evtkModuleFound:
        
        isize = U.shape[0]
        jsize = U.shape[1]
        
        from evtk.hl import imageToVTK
        
        # compelled to deepcopy array
        # apparently slices are not accepted by imageToVTK
        d = U[:,:,ID].reshape(isize,jsize,1).copy()
        p = U[:,:,IP].reshape(isize,jsize,1).copy()
        u = U[:,:,IU].reshape(isize,jsize,1).copy()
        v = U[:,:,IV].reshape(isize,jsize,1).copy()
        
        imageToVTK(filename, cellData = {"rho" : d,
                                         "E"   : p,
                                         "mx"  : u,
                                         "my"  : v})
        
    else:
        # ? try to write a vtk file by hand .... later
        pass
    
# ################################
# EULER2D MAIN
# ################################
if __name__ == '__main__':

    if len(sys.argv) > 1:
        paramFilename = sys.argv[1]
    else:
        paramFilename = './test_implode.ini'

    # parser input parameter file
    Config = ConfigParser.ConfigParser()
    Config.read(paramFilename)
    print Config

    t  = 0.0
    dt = 0.0
    nStep = 0

    # read parameter file
    par = hp.hydroParams(paramFilename)
    par.printConfig()
    
    # create the main data structure : a hydroRun (a changer --> dico)
    parDict = par.getDict()
    hr = hydroRun.hydroRun(parDict)

    # initial condition
    hr.init_condition()

    # initialize time step
    dt = hr.compute_dt(0)
    print "dt = {}".format(dt)

    # initialize boundaries
    hr.make_boundaries(0)
    hr.make_boundaries(1)

    # start computation
    print "Start computation...."
    hydroMonitoring.total_timer.start()

    # Hydrodynamics solver loop
    t     = 0.0
    nStep = 0
    while t < parDict['tEnd'] and nStep < parDict['nStepmax']:
        # output
        hydroMonitoring.io_timer.start()
        if nStep % parDict['nOutput'] == 0:
            print "Output results at time t={0:16.13f} step {1:05d} dt={2:13.10f}".format(t,nStep,dt)
            filename = "U_{0:03d}".format(nStep)
            if nStep % 2 == 0:
                saveVTK(hr.U,  filename)
            else:
                saveVTK(hr.U2, filename)

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
    print "compute     time : {0:5.3f} secondes {1:6.3f}%".format(t_comp,t_comp/t_tot*100)
    print "boundaries  time : {0:5.3f} secondes {1:6.3f}%".format(t_bound,t_bound/t_tot*100)
    print "io          time : {0:5.3f} secondes {1:6.3f}%".format(t_io,t_io/t_tot*100)
    print "Perf             : {0:10.2f} number of cell-updates/s".format(nStep*par.isize*par.jsize/t_tot)

  
