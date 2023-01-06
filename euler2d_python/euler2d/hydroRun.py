# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:06:19 2014

@author: pkestene
"""

import numpy as np

import copy

from . import hydroParam
from . import hydroUtils
from . import hydroMonitoring

ID    = hydroParam.ID
IP    = hydroParam.IP
IU    = hydroParam.IU
IV    = hydroParam.IV

NBVAR = hydroParam.NBVAR
TWO_D = hydroParam.TWO_D

FACE_XMIN = hydroParam.FACE_XMIN
FACE_XMAX = hydroParam.FACE_XMAX
FACE_YMIN = hydroParam.FACE_YMIN
FACE_YMAX = hydroParam.FACE_YMAX

IX    = hydroParam.IX
IY    = hydroParam.IY

BC_DIRICHLET = hydroParam.BC_DIRICHLET
BC_NEUMANN   = hydroParam.BC_NEUMANN
BC_PERIODIC  = hydroParam.BC_PERIODIC
BC_COPY      = hydroParam.BC_COPY

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


########################################################
# `hydroRun` class
########################################################
class hydroRun(object):

    ########################################################
    def __init__(self, iniFilename):

        #print("toto : " + iniFilename)
        #super(hydroRun,self).__init__(iniFilename)

        self.param = hydroParam.hydroParams(iniFilename)

        self.utils = hydroUtils.hydroUtils(self.param)

        NBVAR = hydroParam.NBVAR

        # define workspace
        self.U  = np.zeros((self.param.isize,self.param.jsize,NBVAR))
        self.U2 = np.zeros((self.param.isize,self.param.jsize,NBVAR))
        self.Q  = np.zeros((self.param.isize,self.param.jsize,NBVAR))

        # if we use implementation version 1, we need other arrays
        if self.param.implementationVersion == 1:
            self.Qm_x = np.zeros((self.param.isize,self.param.jsize,NBVAR))
            self.Qm_y = np.zeros((self.param.isize,self.param.jsize,NBVAR))
            self.Qp_x = np.zeros((self.param.isize,self.param.jsize,NBVAR))
            self.Qp_y = np.zeros((self.param.isize,self.param.jsize,NBVAR))

    ########################################################
    def init_condition(self):

        problem = self.param.problem

        if problem == "implode":

            self.init_implode(self.U)
            self.init_implode(self.U2)

        elif problem == "blast":

            self.init_blast(self.U)
            self.init_blast(self.U2)

        else:

            print("Problem : % is not recognized / implemented ...".format(problem))

    ########################################################
    def init_implode(self,U):

        gw = self.param.ghostWidth

        for j in range(0,self.param.jsize):
            for i in range(0,self.param.isize):

                tmp = 1.0*(i-gw)/self.param.nx + 1.0*(j-gw)/self.param.ny
                if tmp>0.5:
                    U[i,j,ID] = 1.0
                    U[i,j,IP] = 1.0/(self.param.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0
                else:
                    U[i,j,ID] = 0.125 #+0.1*np.random.uniform()
                    U[i,j,IP] = 0.14/(self.param.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0

    ########################################################
    def init_blast(self,U):

        gw = self.param.ghostWidth

        radius2 = self.param.radius**2

        center_x = self.param.center_x
        center_y = self.param.center_y

        density_in = self.param.density_in
        density_out = self.param.density_out
        pressure_in = self.param.pressure_in
        pressure_out = self.param.pressure_out

        for j in range(0,self.param.jsize):
            for i in range(0,self.param.isize):

                d2 = (i-center_x)*(i-center_x)+(j-center_y)*(j-center_y)
                if d2<radius2:
                    U[i,j,ID] = density_in
                    U[i,j,IP] = pressure_in/(self.param.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0
                else:
                    U[i,j,ID] = density_out #+0.1*np.random.uniform()
                    U[i,j,IP] = pressure_out/(self.param.gamma0-1.0)
                    U[i,j,IU] = 0.0
                    U[i,j,IV] = 0.0

    ########################################################
    def compute_dt(self,useU):
        """
        Compute time step satisfying CFL condition.

        :param useU: specify which hydrodynamics data array
        :returns: dt time step
        """

        if useU == 0:
            U = self.U
        else:
            U = self.U2

        invDt = 0
        dx = self.param.dx
        dy = self.param.dy

        for j in range(0,self.param.jsize):
            for i in range(0,self.param.isize):
                qLoc, c = self.utils.computePrimitives_ij(U,i,j)
                vx = c + abs(qLoc[IU])
                vy = c + abs(qLoc[IV])

                invDt = max(invDt, vx/dx + vy/dy);

        return self.param.cfl / invDt

    ########################################################
    def compute_primitives(self,U):
        """
        Convert conservative variables to primitive.
        """
        for j in range(0,self.param.jsize):
            for i in range(0,self.param.isize):
                qLoc, c = self.utils.computePrimitives_ij(U,i,j)
                self.Q[i,j,:] = qLoc[:]

    ########################################################
    def make_boundaries(self,useU):
        """
        Fill ghost boundaries.
        """

        if useU == 0:
            U = self.U
        else:
            U = self.U2

        gw     = self.param.ghostWidth
        b_xmin = self.param.boundary_type_xmin
        b_xmax = self.param.boundary_type_xmax
        b_ymin = self.param.boundary_type_ymin
        b_ymax = self.param.boundary_type_ymax

        nx = self.param.nx
        ny = self.param.ny

        imin = self.param.imin
        imax = self.param.imax
        jmin = self.param.jmin
        jmax = self.param.jmax

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

                for j in range(jmin+gw, jmax-gw+1):
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

                for j in range(jmin+gw, jmax-gw+1):
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

                for i in range(imin+gw, imax-gw+1):
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

                for i in range(imin+gw, imax-gw+1):
                    U[i,j,iVar] = U[i,j0,iVar]*sign

    ########################################################
    def godunov_unsplit(self,nStep,dt):
        """
        Wrapper to main routine for performing one time step integration.
        """

        if nStep%2 == 0:
            U  = self.U
            U2 = self.U2
        else:
            U  = self.U2
            U2 = self.U

        self.godunov_unsplit_cpu(U , U2, dt, nStep)

    ########################################################
    def godunov_unsplit_cpu(self, U, U2, dt, nStep):
        """
        This is the main routine for performing one time step integration.
        """

        dtdx  = dt / self.param.dx
        dtdy  = dt / self.param.dy
        isize = self.param.isize
        jsize = self.param.jsize
        gw    = self.param.ghostWidth

        # fill ghost cell in data_in
        hydroMonitoring.boundaries_timer.start()
        self.make_boundaries(nStep%2)
        hydroMonitoring.boundaries_timer.stop()

        # copy U into U2
        U2[:,:,:] = U[:,:,:]

        # main computation
        hydroMonitoring.godunov_timer.start()

        # convert to primitive variables
        self.compute_primitives(U)

        if self.param.implementationVersion==0:

            for j in range(gw, jsize-gw+1):
                for i in range(gw, isize-gw+1):

                    # primitive variables in neighborhood
                    qLoc = np.zeros(NBVAR)
                    qLocN = np.zeros(NBVAR)
                    qNeighbors = np.zeros((2*TWO_D,NBVAR))

                    # get slopes in current cell
                    qLoc[:] = self.Q[i,j,:]

                    qNeighbors[0] = self.Q[i+1,j  ,:]
                    qNeighbors[1] = self.Q[i-1,j  ,:]
                    qNeighbors[2] = self.Q[i  ,j+1,:]
                    qNeighbors[3] = self.Q[i  ,j-1,:]

                    # compute slopes in current cell
                    dqX, dqY = self.utils.slope_unsplit_hydro_2d(qLoc, qNeighbors)

                    ##################################
                    # left interface along X direction
                    ##################################

                    # get primitive variables state vector in
                    # left neighbor along X
                    qLocN[:] = self.Q[i-1,j,:]

                    qNeighbors[0] = self.Q[i  ,j  ,:]
                    qNeighbors[1] = self.Q[i-2,j  ,:]
                    qNeighbors[2] = self.Q[i-1,j+1,:]
                    qNeighbors[3] = self.Q[i-1,j-1,:]

                    # compute slopes in left neighbor along X
                    dqX_n, dqY_n = self.utils.slope_unsplit_hydro_2d(qLocN, qNeighbors)

                    #
                    # Compute reconstructed states at left interface
                    #  along X in current cell
                    #

                    # left interface : right state
                    qright = self.utils.trace_unsplit_hydro_2d_by_direction(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN)

                    # left interface : left state
                    qleft = self.utils.trace_unsplit_hydro_2d_by_direction(qLocN, dqX_n, dqY_n, dtdx, dtdy, FACE_XMAX)

                    flux_x = self.utils.riemann_2d(qleft,qright)

                    ##################################
                    # left interface along Y direction
                    ##################################

                    # get primitive variables state vector in
                    # left neighbor along Y
                    qLocN[:] = self.Q[i,j-1,:]

                    qNeighbors[0] = self.Q[i+1,j-1,:]
                    qNeighbors[1] = self.Q[i-1,j-1,:]
                    qNeighbors[2] = self.Q[i  ,j  ,:]
                    qNeighbors[3] = self.Q[i  ,j-2,:]

                    # compute slopes in current cell
                    dqX_n, dqY_n = self.utils.slope_unsplit_hydro_2d(qLocN, qNeighbors)

                    #
                    # Compute reconstructed states at left interface
                    #  along X in current cell
                    #

                    # left interface : right state
                    qright = self.utils.trace_unsplit_hydro_2d_by_direction(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN)

                    # left interface : left state
                    qleft = self.utils.trace_unsplit_hydro_2d_by_direction(qLocN, dqX_n, dqY_n,dtdx, dtdy, FACE_YMAX)

                    qleft[IU], qleft[IV] = qleft[IV], qleft[IU] # watchout IU, IV permutation

                    qright[IU], qright[IV]  = qright[IV], qright[IU] # watchout IU, IV permutation

                    flux_y = self.utils.riemann_2d(qleft,qright)

                    # swap flux_y components
                    flux_y[IU], flux_y[IV] = flux_y[IV], flux_y[IU]

                    #
                    # update hydro array
                    #
                    U2[i-1,j  ,:] += (-flux_x[:]*dtdx)
                    U2[i  ,j  ,:] += ( flux_x[:]*dtdx)

                    U2[i  ,j-1,:] += (-flux_y[:]*dtdy)
                    U2[i  ,j  ,:] += ( flux_y[:]*dtdy)

        else:
            # TODO
            pass

        hydroMonitoring.godunov_timer.stop()

########################################################
def saveVTK(U, filename):
    """
    Main IO routine. Save U array into a vtk file.
    filename should NOT include .vti suffix
    """

    # use vtk module if available
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

        # compelled to deepcopy array
        # apparently slices are not accepted by imageToVTK
        d = U[:,:,ID].reshape(isize,jsize,1).copy()
        p = U[:,:,IP].reshape(isize,jsize,1).copy()
        u = U[:,:,IU].reshape(isize,jsize,1).copy()
        v = U[:,:,IV].reshape(isize,jsize,1).copy()

        evtk.hl.imageToVTK(filename, cellData = {"rho" : d,
                                         "E"   : p,
                                         "mx"  : u,
                                         "my"  : v})

    else:
        # ? try to write a vtk file by hand .... later
        pass
