# cython: infer_types = True
# cython: profile = False
# cython: boundscheck = False
# cython: wraparound = False
# cython: nonecheck = False
# cython: cdivision = True
# cython: language_level = 3

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:06:19 2014

@author: pkestene
"""


import numpy as np
cimport numpy as np

cimport cython

from . import hydroParam as hp
cimport euler2d.hydroUtils as hydroUtils

from . import hydroMonitoring


# test if vtk module is available
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from vtk.util import numpy_support
    vtkModuleFound = True
except ImportError:
    vtkModuleFound = False

# test if tvtk module is available
# Example of use:
# see http://docs.enthought.com/mayavi/mayavi/auto/example_datasets.html
# see http://stackoverflow.com/questions/20035620/save-data-to-vtk-using-python-and-tvtk-with-more-than-one-vector-field
try:
    from tvtk.api import tvtk
    from tvtk.api import write_data as tvtk_write_data
    tvtkModuleFound = True
except ImportError:
    tvtkModuleFound = False

# test if pyEVTK is available
# See https://bitbucket.org/pauloh/pyevtk
try:
    import evtk
    evtkModuleFound = True
except:
    evtkModuleFound = False


cdef extern from "math.h":
    bint isnan(double x)
    double fmin(double x, double y)
    double fmax(double x, double y)
    double sqrt(double x)
    double fabs(double x)
    double copysign(double x, double y)

cdef int ID    = hp.ID
cdef int IP    = hp.IP
cdef int IU    = hp.IU
cdef int IV    = hp.IV

cdef int NBVAR = hp.NBVAR

cdef int FACE_XMIN = hp.FACE_XMIN
cdef int FACE_XMAX = hp.FACE_XMAX
cdef int FACE_YMIN = hp.FACE_YMIN
cdef int FACE_YMAX = hp.FACE_YMAX

cdef int IX    = hp.IX
cdef int IY    = hp.IY

cdef int BC_DIRICHLET = hp.BC_DIRICHLET
cdef int BC_NEUMANN   = hp.BC_NEUMANN
cdef int BC_PERIODIC  = hp.BC_PERIODIC
cdef int BC_COPY      = hp.BC_COPY

########################################################
# `hydroRun` class
########################################################
cdef class hydroRun:
    """
    Main class to perform Euler simulation.
    """

    cdef dict param
    cdef hydroUtils.hydroUtils utils

    cdef public int isize, jsize, implementationVersion
    cdef public str problem
    cdef public int gw, nx, ny, imin, imax, jmin, jmax
    cdef public double dx, dy, gamma0, cfl
    cdef public int boundary_type_xmin, boundary_type_xmax
    cdef public int boundary_type_ymin, boundary_type_ymax
    cdef public np.ndarray U
    cdef public np.ndarray U2
    cdef np.ndarray Q
    cdef np.ndarray Qm_x
    cdef np.ndarray Qm_y
    cdef np.ndarray Qp_x
    cdef np.ndarray Qp_y

    ########################################################
    def __cinit__(self, dict param):

        self.param = param

        self.utils = hydroUtils.hydroUtils(self.param)

        print(param)
        self.isize = param['isize']
        self.jsize = param['jsize']
        self.implementationVersion = param['implementationVersion']
        self.problem = param['problem']
        self.gw = param['ghostWidth']
        self.nx = param['nx']
        self.ny = param['ny']
        self.imin = param['imin']
        self.imax = param['imax']
        self.jmin = param['jmin']
        self.jmax = param['jmax']
        self.dx = param['dx']
        self.dy = param['dy']
        self.gamma0 =param['gamma0']
        self.cfl = param['cfl']
        self.boundary_type_xmin = param['boundary_type_xmin']
        self.boundary_type_xmax = param['boundary_type_xmax']
        self.boundary_type_ymin = param['boundary_type_ymin']
        self.boundary_type_ymax = param['boundary_type_ymax']


        # define workspace
        self.U  = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)
        self.U2 = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)
        self.Q  = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)

        # if we use implementation version 1, we need other arrays
        if self.implementationVersion == 1:
            self.Qm_x = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)
            self.Qm_y = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)
            self.Qp_x = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)
            self.Qp_y = np.zeros((self.isize,self.jsize,NBVAR), dtype=np.double)

    ########################################################
    def init_condition(self):

        if self.problem == "implode":

            self.init_implode(self.U)
            self.init_implode(self.U2)

        elif self.problem == "blast":

            self.init_blast(self.U)
            self.init_blast(self.U2)

        else:

            print("Problem : % is not recognized / implemented ...".format(self.problem))

    ########################################################
    cdef init_implode(self, np.ndarray[double, ndim=3] U):
        """
        Initial condition for implode test.
        """
        cdef int i, j
        cdef double tmp

        for j in range(0,self.jsize):
            for i in range(0,self.isize):

                tmp = 1.0*(i-self.gw)/self.nx + 1.0*(j-self.gw)/self.ny
                if tmp>0.5:
                    U[i,j,ID] = 1.0
                    U[i,j,IP] = 1.0/(self.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0
                else:
                    U[i,j,ID] = 0.125 #+0.1*np.random.uniform()
                    U[i,j,IP] = 0.14/(self.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0

    ########################################################
    cdef init_blast(self, np.ndarray[double, ndim=3] U):
        """
        Initial condition for implode test.
        """
        cdef int i, j
        cdef double radius2 = self.param['radius']**2
        cdef double d2

        cdef double center_x = self.param['center_x']
        cdef double center_y = self.param['center_y']
        cdef double density_in = self.param['density_in']
        cdef double density_out = self.param['density_out']
        cdef double pressure_in = self.param['pressure_in']
        cdef double pressure_out = self.param['pressure_out']

        for j in range(0,self.jsize):
            for i in range(0,self.isize):

                d2 = (i-center_x)*(i-center_x)+(j-center_y)*(j-center_y)
                if d2<radius2:
                    U[i,j,ID] = density_in
                    U[i,j,IP] = pressure_in/(self.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0
                else:
                    U[i,j,ID] = density_out #+0.1*np.random.uniform()
                    U[i,j,IP] = pressure_out/(self.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0

    ########################################################
    cpdef double compute_dt(self, int useU):
        """
        Compute time step satisfying CFL condition.

        :param useU: specify which hydrodynamics data array
        :returns: dt time step
        """

        cdef double[:,:,:] U
        cdef double invDt, dx, dy, vx, vy, c
        cdef double[4] uLoc
        cdef double[4] qLoc

        if useU == 0:
            U = self.U
        else:
            U = self.U2

        invDt = 0.0
        dx = self.dx
        dy = self.dy

        for j in range(0,self.jsize):
            for i in range(0,self.isize):

                for ivar in range(NBVAR):
                    uLoc[ivar] = U[i,j,ivar]
                self.utils.computePrimitives(uLoc,qLoc, &c)
                vx = c + fabs(qLoc[IU])
                vy = c + fabs(qLoc[IV])

                invDt = fmax(invDt, vx/dx + vy/dy);

        return self.cfl / invDt

    ########################################################
    def make_boundaries(self, int useU):
        """
        Fill ghost boundaries.
        """

        cdef double[:,:,:] U
        cdef int b_xmin, b_xmax, b_ymin, b_ymax
        cdef int nx, ny, i0, j0, i, j, iVar, gw
        cdef int imin, imax, jmin, jmax
        cdef double sign

        if useU == 0:
            U = self.U
        else:
            U = self.U2

        b_xmin = self.boundary_type_xmin
        b_xmax = self.boundary_type_xmax
        b_ymin = self.boundary_type_ymin
        b_ymax = self.boundary_type_ymax

        nx = self.nx
        ny = self.ny

        imin = self.imin
        imax = self.imax
        jmin = self.jmin
        jmax = self.jmax

        gw = self.gw

        # boundary xmin
        for iVar in range(NBVAR):
            for i in range(gw):
                sign = 1.0
                if   b_xmin == BC_DIRICHLET:
                    i0 = 2*gw-1-i
                    if iVar==IU:
                        sign = -1.0
                elif b_xmin == BC_NEUMANN:
                    i0 = gw
                else: # periodic
                    i0 = nx+i

                for j in range(jmin, jmax):
                    U[i,j,iVar] = U[i0,j,iVar]*sign


        # boundary xmax
        for iVar in range(NBVAR):
            for i in range (nx+gw, nx+2*gw):
                sign = 1.0
                if b_xmax == BC_DIRICHLET:
                    i0 = 2*nx + 2*gw-1-i
                    if iVar==IU:
                        sign = -1.0
                elif b_xmax == BC_NEUMANN:
                    i0 = nx+gw-1
                else:  # periodic
                    i0 = i-nx

                for j in range(jmin, jmax):
                    U[i,j,iVar] = U[i0,j,iVar]*sign

        # boundary ymin
        for iVar in range(NBVAR):
            for j in range(gw):
                sign = 1.0
                if b_ymin == BC_DIRICHLET:
                    j0 = 2*gw-1-j
                    if iVar==IV:
                        sign = -1.0
                elif b_ymin == BC_NEUMANN:
                    j0 = gw
                else:  # periodic
                    j0 = ny+j

                for i in range(imin, imax):
                    U[i,j,iVar] =  U[i,j0,iVar]*sign

        # boundary ymax
        for iVar in range(NBVAR):
            for j in range(ny+gw, ny+2*gw):
                sign = 1.0
                if b_ymax == BC_DIRICHLET:
                    j0 = 2*ny+2*gw-1-j
                    if iVar==IV:
                        sign = -1.0
                elif b_ymax == BC_NEUMANN:
                    j0 = ny+gw-1
                else:  # periodic
                    j0 = j-ny

                for i in range(imin, imax):
                    U[i,j,iVar] = U[i,j0,iVar]*sign

    ########################################################
    def convertToPrimitive(self,
                           np.ndarray[double, ndim=3] U,
                           np.ndarray[double, ndim=3] Q):

        cdef int i,j
        cdef int isize, jsize, gw

        cdef double[4] qLoc
        cdef double[4] uLoc
        cdef double c
        cdef int ivar

        isize = self.isize
        jsize = self.jsize
        gw    = self.gw

        for j in range(0, jsize):
            for i in range(0, isize):

                # get local conserved variables
                for ivar in range(NBVAR):
                    uLoc[ivar] = U[i,j,ivar]

                # convert to primitive variables
                self.utils.computePrimitives(uLoc,qLoc,&c)

                # write back primitive variables to Q array
                for ivar in range(NBVAR):
                    Q[i,j,ivar] = qLoc[ivar]

    ########################################################
    def computeSlopesAndTrace(self,
                              np.ndarray[double, ndim=3] Q,
                              np.ndarray[double, ndim=3] Qm_x,
                              np.ndarray[double, ndim=3] Qm_y,
                              np.ndarray[double, ndim=3] Qp_x,
                              np.ndarray[double, ndim=3] Qp_y,
                              double dt
                              ):

        cdef int i,j, ivar
        cdef int isize, jsize
        cdef double dtdx, dtdy

        # primitive variables
        cdef double[4] qLoc
        cdef double[4][4] qNeighbors

        # slopes
        cdef double[4] dqX
        cdef double[4] dqY

        # trace
        cdef double[2][4] qm
        cdef double[2][4] qp

        dtdx = dt / self.dx
        dtdy = dt / self.dy
        isize = self.isize
        jsize = self.jsize

        for j in range(1, jsize-1):
            for i in range(1, isize-1):

                # retrieve primitive variables in neighborhood
                for ivar in range(NBVAR):
                    qLoc[ivar] = Q[i  ,j  ,ivar]
                    qNeighbors[0][ivar] = Q[i+1,j  ,ivar]
                    qNeighbors[1][ivar] = Q[i-1,j  ,ivar]
                    qNeighbors[2][ivar] = Q[i  ,j+1,ivar]
                    qNeighbors[3][ivar] = Q[i  ,j-1,ivar]

                # compute slopes: dqX and dqY
                self.utils.slope_unsplit_hydro_2d(qLoc, qNeighbors, dqX, dqY)

                # compute trace (qm and qp)
                self.utils.trace_unsplit_hydro_2d(qLoc, dqX, dqY, dtdx, dtdy, qm, qp)

                # gravity predictor (TODO)

                # store qm, qp : only what is really needed
                for ivar in range(NBVAR):
                    Qm_x[i,j,ivar] = qm[0][ivar]
                    Qp_x[i,j,ivar] = qp[0][ivar]
                    Qm_y[i,j,ivar] = qm[1][ivar]
                    Qp_y[i,j,ivar] = qp[1][ivar]


    ########################################################
    def computeFluxesAndUpdate(self,
                               np.ndarray[double, ndim=3] U,
                               np.ndarray[double, ndim=3] U2,
                               np.ndarray[double, ndim=3] Qm_x,
                               np.ndarray[double, ndim=3] Qm_y,
                               np.ndarray[double, ndim=3] Qp_x,
                               np.ndarray[double, ndim=3] Qp_y,
                               double dt
                               ):

        cdef int i,j, ivar
        cdef int isize, jsize, gw
        cdef double dtdx, dtdy

        cdef double[4] qleft
        cdef double[4] qright
        cdef double[4] flux_x
        cdef double[4] flux_y

        dtdx = dt / self.dx
        dtdy = dt / self.dy
        isize = self.isize
        jsize = self.jsize
        gw = self.gw

        for j in range(gw, jsize-gw+1):
            for i in range(gw, isize-gw+1):


                #
                # solve Riemann problem at X-interfaces
                #
                for ivar in range(NBVAR):
                    qleft[ivar]  = Qm_x[i-1,j  ,ivar]
                    qright[ivar] = Qp_x[i  ,j  ,ivar]

                self.utils.riemann_2d(qleft,qright,flux_x)

                #
                # solve Riemann problem at Y-interfaces
                #
                for ivar in range(NBVAR):
                    qleft[ivar]  = Qm_y[i  ,j-1,ivar]
                    qright[ivar] = Qp_y[i  ,j  ,ivar]

                # watchout IU, IV permutation
                qleft[IU], qleft[IV] = qleft[IV], qleft[IU]
                qright[IU], qright[IV]  = qright[IV], qright[IU]

                self.utils.riemann_2d(qleft,qright,flux_y)
                # swap flux_y components
                flux_y[IU], flux_y[IV] = flux_y[IV], flux_y[IU]

                #
                # update hydro array
                #
                for ivar in range(NBVAR):
                    U2[i-1,j  ,ivar] += (-flux_x[ivar]*dtdx)
                    U2[i  ,j  ,ivar] += ( flux_x[ivar]*dtdx)

                    U2[i  ,j-1,ivar] += (-flux_y[ivar]*dtdy)
                    U2[i  ,j  ,ivar] += ( flux_y[ivar]*dtdy)


    ########################################################
    def godunov_unsplit(self, int nStep, double dt):
        """
        Wrapper to main routine for performing one time step integration.
        """

        if nStep%2 == 0:
            self.godunov_unsplit_cpu(self.U , self.U2, dt, nStep)
        else:
            self.godunov_unsplit_cpu(self.U2, self.U , dt, nStep)

    ########################################################
    cdef godunov_unsplit_cpu(self,
                             np.ndarray[double, ndim=3] U,
                             np.ndarray[double, ndim=3] U2,
                             double dt,
                             int nStep):
        """
        This is the main routine for performing one time step integration.
        """

        cdef double dtdx, dtdy
        cdef int isize, jsize, gw
        cdef double[4] uLoc
        cdef double[4] qLoc
        cdef double[4] qLocN
        cdef double[4][4] qNeighbors

        cdef double[4] dqX
        cdef double[4] dqY
        cdef double[4] dqX_n
        cdef double[4] dqY_n

        cdef double[4] qleft
        cdef double[4] qright
        cdef double[4] flux_x
        cdef double[4] flux_y

        cdef double[3][4] qm_x
        cdef double[3][4] qm_y
        cdef double[3][4] qp_x
        cdef double[3][4] qp_y


        cdef double[2][4] qm
        cdef double[2][4] qp

        cdef double c, cPlus, cMinus
        cdef int i, j, k, ii, jj, pos
        cdef int ivar

        cdef np.ndarray[double, ndim=3] Q = self.Q

        dtdx  = dt / self.dx
        dtdy  = dt / self.dy
        isize = self.isize
        jsize = self.jsize
        gw    = self.gw

        # fill ghost cell in data_in
        hydroMonitoring.boundaries_timer.start()
        self.make_boundaries(nStep%2)
        hydroMonitoring.boundaries_timer.stop()

        # copy U into U2
        U2[:,:,:] = U[:,:,:]
        # for i in range(isize):
        #     for j in range(jsize):
        #         for k in range(NBVAR):
        #             U2[i,j,k] = U[i,j,k]

        # main computation
        hydroMonitoring.godunov_timer.start()

        # convert to primitive variables
        self.convertToPrimitive(U, Q)

        if self.implementationVersion==0:

            for j in range(gw, jsize-gw+1):
                for i in range(gw, isize-gw+1):

                    # compute slopes in current cell
                    for ivar in range(NBVAR):
                        qLoc[ivar] = Q[i  ,j  , ivar]
                        qNeighbors[0][ivar] = Q[i+1,j  ,ivar]
                        qNeighbors[1][ivar] = Q[i-1,j  ,ivar]
                        qNeighbors[2][ivar] = Q[i  ,j+1,ivar]
                        qNeighbors[3][ivar] = Q[i  ,j-1,ivar]

                    # compute slopes in current cell
                    self.utils.slope_unsplit_hydro_2d(qLoc, qNeighbors, dqX, dqY)

                    ##################################
                    # left interface along X direction
                    ##################################

                    # compute slopes in left neighbor along X
                    for ivar in range(NBVAR):
                        qLocN[ivar] = Q[i-1,j  , ivar]
                        qNeighbors[0][ivar] = Q[i  ,j  ,ivar]
                        qNeighbors[1][ivar] = Q[i-2,j  ,ivar]
                        qNeighbors[2][ivar] = Q[i-1,j+1,ivar]
                        qNeighbors[3][ivar] = Q[i-1,j-1,ivar]

                    # compute slopes in neighbor
                    self.utils.slope_unsplit_hydro_2d(qLocN, qNeighbors, dqX_n, dqY_n)

                    # left interface : right state
                    self.utils.trace_unsplit_hydro_2d_by_direction(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright)

                    # left interface : left state
                    self.utils.trace_unsplit_hydro_2d_by_direction(qLocN, dqX_n, dqY_n, dtdx, dtdy, FACE_XMAX, qleft)

                    # compute riemann problem for X interface
                    self.utils.riemann_2d(qleft,qright,flux_x)

                    ##################################
                    # left interface along Y direction
                    ##################################

                    # compute slopes in left neighbor along Y
                    for ivar in range(NBVAR):
                        qLocN[ivar] = Q[i  ,j-1, ivar]
                        qNeighbors[0][ivar] = Q[i+1,j-1,ivar]
                        qNeighbors[1][ivar] = Q[i-1,j-1,ivar]
                        qNeighbors[2][ivar] = Q[i  ,j  ,ivar]
                        qNeighbors[3][ivar] = Q[i  ,j-2,ivar]

                    # compute slopes in neighbor
                    self.utils.slope_unsplit_hydro_2d(qLocN, qNeighbors, dqX_n, dqY_n)

                    # left interface : right state
                    self.utils.trace_unsplit_hydro_2d_by_direction(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright)

                    # left interface : left state
                    self.utils.trace_unsplit_hydro_2d_by_direction(qLocN, dqX_n, dqY_n, dtdx, dtdy, FACE_YMAX, qleft)

                    # watchout IU, IV permutation
                    qleft[IU], qleft[IV] = qleft[IV], qleft[IU]
                    qright[IU], qright[IV] = qright[IV], qright[IU]

                    # compute riemann problem for Y interface
                    self.utils.riemann_2d(qleft,qright,flux_y)

                    # swap flux_y components
                    flux_y[IU], flux_y[IV] = flux_y[IV], flux_y[IU]

                    #
                    # update hydro array
                    #
                    for ivar in range(NBVAR):
                        U2[i-1,j  ,ivar] += (-flux_x[ivar]*dtdx)
                        U2[i  ,j  ,ivar] += ( flux_x[ivar]*dtdx)

                        U2[i  ,j-1,ivar] += (-flux_y[ivar]*dtdy)
                        U2[i  ,j  ,ivar] += ( flux_y[ivar]*dtdy)

        elif self.implementationVersion == 1:

            # compute slopes and perform trace
            self.computeSlopesAndTrace(self.Q,
                                       self.Qm_x,
                                       self.Qm_y,
                                       self.Qp_x,
                                       self.Qp_y,
                                       dt)

            # compute fluxes (using Riemann solver at cell interface)
            # and update (time integration)
            self.computeFluxesAndUpdate(U,
                                        U2,
                                        self.Qm_x,
                                        self.Qm_y,
                                        self.Qp_x,
                                        self.Qp_y,
                                        dt)

        else:
            # unknow version
            pass

        hydroMonitoring.godunov_timer.stop()

########################################################
def saveVTK(U, filename):
    """
    Main IO routine. Save U array into a vtk file.
    filename should NOT include .vti suffix
    """

    # enum for hydro variables (taken from hydroParam)
    ID = hp.ID
    IP = hp.IP
    IU = hp.IU
    IV = hp.IV

    # filename without suffix (.vti)

    # use tvtk module if available
    if vtkModuleFound:
        # create an imageData
        import vtk.util.numpy_support as numpy_support

        img = vtk.vtkImageData()
        img.SetOrigin(0, 0, 0)
        img.SetSpacing(1, 1, 1)
        img.SetDimensions(U[:,:,ID].shape[0]+1, U[:,:,ID].shape[1]+1, 1)

        vtk_data_ID = numpy_support.numpy_to_vtk(num_array=U[:,:,ID].flatten(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data_ID.SetName("rho")
        img.GetCellData().AddArray(vtk_data_ID)

        vtk_data_IP = numpy_support.numpy_to_vtk(num_array=U[:,:,IP].flatten(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data_IP.SetName("E")
        img.GetCellData().AddArray(vtk_data_IP)

        vtk_data_IU = numpy_support.numpy_to_vtk(num_array=U[:,:,IU].flatten(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data_IU.SetName("mx")
        img.GetCellData().AddArray(vtk_data_IU)

        vtk_data_IV = numpy_support.numpy_to_vtk(num_array=U[:,:,IV].flatten(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data_IV.SetName("my")
        img.GetCellData().AddArray(vtk_data_IV)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename+'.vti')
        writer.SetInputData(img)
        writer.Update()

    elif tvtkModuleFound:
        # create an imageData
        i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))

        # add density field
        i.cell_data.scalars = U[:,:,ID].ravel()
        i.cell_data.scalars.name = 'rho'
        i.dimensions = (U[:,:,ID].shape[0]+1, U[:,:,ID].shape[1]+1, 1)

        # add total energy
        i.cell_data.add_array(U[:,:,IP].ravel())
        i.cell_data.get_array(1).name = 'E'
        i.cell_data.update()

        # add velocity
        i.cell_data.add_array(U[:,:,IU].ravel())
        i.cell_data.get_array(2).name = 'mx'
        i.cell_data.update()

        i.cell_data.add_array(U[:,:,IV].ravel())
        i.cell_data.get_array(3).name = 'my'
        i.cell_data.update()

        # actual write data on disk
        tvtk_write_data(i, filename)

        #writer = tvtk.vtkXMLImageDataWriter()
        #writer.SetFileName(filename)

    # use evtk module if available
    elif evtkModuleFound:

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
