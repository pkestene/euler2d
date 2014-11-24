# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:06:19 2014

@author: pkestene
"""

import numpy as np

import copy

import hydroParam
import hydroUtils
import hydroMonitoring

ID    = hydroParam.ID
IP    = hydroParam.IP
IU    = hydroParam.IU
IV    = hydroParam.IV

NBVAR = hydroParam.NBVAR

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


########################################################
# `hydroRun` class
########################################################
class hydroRun(object):
    
    ########################################################
    def __init__(self, iniFilename):
        
        #print "toto : " + iniFilename
        #super(hydroRun,self).__init__(iniFilename)
        
        self.param = hydroParam.hydroParams(iniFilename)

        self.utils = hydroUtils.hydroUtils(self.param)

        NBVAR = hydroParam.NBVAR

        # define workspace
        self.U  = np.zeros((self.param.isize,self.param.jsize,NBVAR))
        self.U2 = np.zeros((self.param.isize,self.param.jsize,NBVAR))

        # if we use implementation version 1, we need other arrays
        if self.param.implementationVersion == 1:
            self.Q    = np.zeros((self.param.isize,self.param.jsize,NBVAR))
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

            print "Problem : % is not recognized / implemented ...".format(problem)

    ########################################################
    def init_implode(self,U):

        gw = self.param.ghostWidth

        for j in xrange(0,self.param.jsize):
            for i in xrange(0,self.param.isize):

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
        
        for j in xrange(0,self.param.jsize):
            for i in xrange(0,self.param.isize):

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

        for j in xrange(0,self.param.jsize):
            for i in xrange(0,self.param.isize):
                qLoc, c = self.utils.computePrimitives_ij(U,i,j)
                vx = c + abs(qLoc[IU])
                vy = c + abs(qLoc[IV])

                invDt = max(invDt, vx/dx + vy/dy);

        return self.param.cfl / invDt

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
        for iVar in xrange(NBVAR):
            for i in xrange(gw):
                sign = 1.0
                if   b_xmin == BC_DIRICHLET:
                    i0 = 2*gw-1-i
                    if iVar==IU:
                        sign = -1.0
                elif b_xmin == BC_NEUMANN:
                    i0 = gw
                else: # periodic
                    i0 = nx+i

                for j in xrange(jmin+gw, jmax-gw+1):
                    U[i,j,iVar] = U[i0,j,iVar]*sign


        # boundary xmax
        for iVar in xrange(NBVAR):
            for i in xrange (nx+gw, nx+2*gw):
                sign = 1.0
                if b_xmax == BC_DIRICHLET:
                    i0 = 2*nx + 2*gw-1-i
                    if iVar==IU:
                        sign = -1.0
                elif b_xmax == BC_NEUMANN:
                    i0 = nx+gw-1
                else:  # periodic
                    i0 = i-nx
                
                for j in xrange(jmin+gw, jmax-gw+1):
                    U[i,j,iVar] = U[i0,j,iVar]*sign
  
        # boundary ymin
        for iVar in xrange(NBVAR):
            for j in xrange(gw):
                sign = 1.0
                if b_ymin == BC_DIRICHLET:
                    j0 = 2*gw-1-j
                    if iVar==IV:
                        sign = -1.0
                elif b_ymin == BC_NEUMANN:
                    j0 = gw
                else:  # periodic
                    j0 = ny+j
      
                for i in xrange(imin+gw, imax-gw+1):
                    U[i,j,iVar] =  U[i,j0,iVar]*sign
        
        # boundary ymax
        for iVar in xrange(NBVAR):
            for j in xrange(ny+gw, ny+2*gw):
                sign = 1.0
                if b_ymax == BC_DIRICHLET:
                    j0 = 2*ny+2*gw-1-j
                    if iVar==IV:
                        sign = -1.0
                elif b_ymax == BC_NEUMANN:
                    j0 = ny+gw-1
                else:  # periodic
                    j0 = j-ny
      
                for i in xrange(imin+gw, imax-gw+1):
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

        if self.param.implementationVersion==0:

            for j in xrange(gw, jsize-gw+1):
                for i in xrange(gw, isize-gw+1):
	
                    qm_x = np.zeros((3,NBVAR))
                    qm_y = np.zeros((3,NBVAR))
                    qp_x = np.zeros((3,NBVAR))
                    qp_y = np.zeros((3,NBVAR))

                    # compute qm, qp for the 1+2 positions
                    for pos in xrange (3):
	  
                        ii=copy.deepcopy(i)
                        jj=copy.deepcopy(j)
                        if pos==1:
                            ii -= 1
                        if pos==2:
                            jj -= 1

                        qNeighbors = np.zeros((4,NBVAR))
     
                        qLoc,   c             = self.utils.computePrimitives_ij(U, ii  , jj  )
                        qNeighbors[0], cPlus  = self.utils.computePrimitives_ij(U, ii+1, jj  )
                        qNeighbors[1], cMinus = self.utils.computePrimitives_ij(U, ii-1, jj  )
                        qNeighbors[2], cPlus  = self.utils.computePrimitives_ij(U, ii  , jj+1)
                        qNeighbors[3], cMinus = self.utils.computePrimitives_ij(U, ii  , jj-1)
	  
                        # compute qm, qp
                        qm, qp = self.utils.trace_unsplit_2d(qLoc, qNeighbors, c, dtdx, dtdy)
	  
                        # store qm, qp
                        qm_x[pos] = qm[0]
                        qp_x[pos] = qp[0]
                        qm_y[pos] = qm[1]
                        qp_y[pos] = qp[1]

                    # Solve Riemann problem at X-interfaces and compute X-fluxes
                    qleft   = qm_x[1]
                    qright  = qp_x[0]

                    flux_x = self.utils.riemann_2d(qleft,qright)

                    # Solve Riemann problem at Y-interfaces and compute Y-fluxes
                    qleft   = qm_y[2]
                    qleft[IU], qleft[IV] = qleft[IV], qleft[IU] # watchout IU, IV permutation
	  
                    qright  = qp_y[0]
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
    filename should include .vti suffix
    """

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
