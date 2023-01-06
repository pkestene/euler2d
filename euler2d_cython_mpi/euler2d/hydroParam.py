# -*- coding: utf-8 -*-
"""
Define class hydroParams.

"""
from mpi4py import MPI

#from enum import Enum
#
#class ComponentIndex(Enum):
#    ID=0  # ID Density field index
#    IP=1  # IP Pressure/Energy field index
#    IU=2  # X velocity / momentum index
#    IV=3  # Y velocity / momentum index

ID=0  # ID Density field index
IP=1  # IP Pressure/Energy field index
IU=2  # X velocity / momentum index
IV=3  # Y velocity / momentum index

NBVAR = 4

TWO_D = 2

# identifying direction
IX=0
IY=1

# identifying directions for reconstructed state on cell faces
FACE_XMIN=0
FACE_XMAX=1
FACE_YMIN=2
FACE_YMAX=3

# type of boundary condition (note that BC_COPY is only used in the
# MPI version for inside boundary)
# enum BoundaryConditionType
BC_UNDEFINED = 0
BC_DIRICHLET = 1   # reflecting border condition
BC_NEUMANN   = 2   # absorbing border condition
BC_PERIODIC  = 3   # periodic border condition
BC_COPY      = 4   # only used in MPI parallelized version


########################################################
# `hydroParams` class
########################################################
class hydroParams(object):

    ########################################################
    def __init__(self, iniFilename):

        self._parseIni(iniFilename)
        self._setParamFromFile()
        self._setParamOther()

    ########################################################
    def _parseIni(self, iniFilename):

        import configparser

        # parser input parameter file
        self.Config = configparser.ConfigParser()
        self.Config.read(iniFilename)

    ########################################################
    def _setParamFromFile(self):
        """
        Parse ini file, and set parameters.
        """

        self.tEnd     = self.Config['RUN'].getfloat('tEnd')
        self.nStepmax = self.Config['RUN'].getint('nStepmax')
        self.nOutput  = self.Config['RUN'].getint('nOutput')

        self.nx       = self.Config['MESH'].getint('nx')
        self.ny       = self.Config['MESH'].getint('ny')

        self.boundary_type_xmin = self.Config['MESH'].getint('boundary_type_xmin')
        self.boundary_type_xmax = self.Config['MESH'].getint('boundary_type_xmax')
        self.boundary_type_ymin = self.Config['MESH'].getint('boundary_type_ymin')
        self.boundary_type_ymax = self.Config['MESH'].getint('boundary_type_ymax')

        # MPI parameters
        # mx,my : number of subdomains along each direction
        # iProc, jProc : cartesian coordinate of subdomain in current MPI proc
        self.mx = self.Config['MESH'].getint('mx')
        self.my = self.Config['MESH'].getint('my')
        #self.iProc = 0
        #self.jProc = 0

        self.gamma0        = self.Config['HYDRO'].getfloat('gamma0')
        self.cfl           = self.Config['HYDRO'].getfloat('cfl')
        self.niter_riemann = self.Config['HYDRO'].getint('niter_riemann')
        self.iorder        = self.Config['HYDRO'].getint('iorder')
        self.slope_type    = self.Config['HYDRO'].getint('slope_type')
        self.problem       = self.Config['HYDRO'].get('problem','implode')

        # parse problem specific parameters
        if self.problem == 'blast':

            if self.Config.has_section('blast'):

                if self.Config['blast'].has_option('radius'):
                    self.radius = self.Config['blast'].getfloat('radius')
                else:
                    self.radius = 1.0

                if self.Config['blast'].has_option('center_x'):
                    self.center_x = self.Config['blast'].getfloat('center_x')
                else:
                    self.center_x = self.nx/2.0


                if self.Config['blast'].has_option('center_y'):
                    self.center_y = self.Config['blast'].getfloat('center_y')
                else:
                    self.center_y = self.ny/2.0


                if self.Config['blast'].has_option('density_in'):
                    self.density_in = self.Config.getfloat('density_in')
                else:
                    self.density_in = 1.0


                if self.Config['blast'].has_option('density_out'):
                    self.density_out = self.Config.getfloat('density_out')
                else:
                    self.density_out = 1.0


                if self.Config['blast'].has_option('pressure_in'):
                    self.pressure_in = self.Config.getfloat('pressure_in')
                else:
                    self.pressure_in = 10.0


                if self.Config['blast'].has_option('pressure_out'):
                    self.pressure_out = self.Config.getfloat('pressure_out')
                else:
                    self.pressure_out = 0.1


            else:
                print('Error: param file should have section named \'blast\'')

        self.implementationVersion  = self.Config['OTHER'].getint('implementationVersion')

    ########################################################
    def _setParamOther(self):

        self.ghostWidth = 2

        self.imin = 0
        self.jmin = 0
        self.imax = self.nx - 1 + 2*self.ghostWidth
        self.jmax = self.ny - 1 + 2*self.ghostWidth

        self.isize = self.imax - self.imin + 1
        self.jsize = self.jmax - self.jmin + 1

        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0

        # take care that dx must use global resolution
        self.dx = (self.xmax - self.xmin) / (self.nx * self.mx)
        self.dy = (self.ymax - self.ymin) / (self.ny * self.my)

        # MPI grid : myRank = iProc + mx*jProc
        myRank = MPI.COMM_WORLD.Get_rank()
        self.jProc = myRank//self.mx
        self.iProc = myRank-self.mx*self.jProc

        self.smallc  = 1e-7
        self.smallr  = 1e-7
        self.smallp  = self.smallc * self.smallc / self.gamma0
        self.smallpp = self.smallr * self.smallp
        self.gamma6  = (self.gamma0 + 1.0)/(2.0 * self.gamma0)


    ########################################################
    def printConfig(self):

        print("##########################")
        print("Simulation run parameters:")
        print("##########################")
        print("nx         : {}".format(self.nx))
        print("ny         : {}".format(self.ny))
        print("mx         : {}".format(self.mx))
        print("my         : {}".format(self.my))
        print("iProc      : {}".format(self.iProc))
        print("jProc      : {}".format(self.jProc))
        print("Global size x : {}".format(self.nx*self.mx))
        print("Global size y : {}".format(self.ny*self.my))
        print("dx         : {}".format(self.dx))
        print("dy         : {}".format(self.dy))
        print("imin       : {}".format(self.imin))
        print("imax       : {}".format(self.imax))
        print("jmin       : {}".format(self.jmin))
        print("jmax       : {}".format(self.jmax))
        print("nStepmax   : {}".format(self.nStepmax))
        print("tEnd       : {}".format(self.tEnd))
        print("nOutput    : {}".format(self.nOutput))
        print("gamma0     : {}".format(self.gamma0))
        print("cfl        : {}".format(self.cfl))
        print("smallr     : {}".format(self.smallr))
        print("smallc     : {}".format(self.smallc))
        print("iorder     : {}".format(self.iorder))
        print("slope_type : {}".format(self.slope_type))
        print("problem    : {}".format(self.problem))


    ########################################################
    def getDict(self):
        """
        Build a dictionnary with parameters.
        Only useful for cython: avoid using ConfigParser in cython.
        """
        dico={}

        dico['ghostWidth']=2
        dico['implementationVersion']=self.implementationVersion
        dico['nx']=self.nx
        dico['ny']=self.ny
        dico['mx']=self.mx
        dico['iProc']=self.iProc
        dico['jProc']=self.jProc
        dico['my']=self.my
        dico['dx']=self.dx
        dico['dy']=self.dy
        dico['imin']=self.imin
        dico['imax']=self.imax
        dico['jmin']=self.jmin
        dico['jmax']=self.jmax
        dico['isize']=self.isize
        dico['jsize']=self.jsize
        dico['xmin']=self.xmin
        dico['xmax']=self.xmax
        dico['ymin']=self.ymin
        dico['ymax']=self.ymax
        dico['nStepmax']=self.nStepmax
        dico['tEnd']=self.tEnd
        dico['nOutput']=self.nOutput
        dico['gamma0']=self.gamma0
        dico['gamma6']=self.gamma6
        dico['cfl']=self.cfl
        dico['smallr']=self.smallr
        dico['smallc']=self.smallc
        dico['smallp']=self.smallp
        dico['smallpp']=self.smallpp
        dico['iorder']=self.iorder
        dico['slope_type']=self.slope_type
        dico['problem']=self.problem
        dico['riemannSolver']='hllc'
        dico['boundary_type_xmin']=self.boundary_type_xmin
        dico['boundary_type_xmax']=self.boundary_type_xmax
        dico['boundary_type_ymin']=self.boundary_type_ymin
        dico['boundary_type_ymax']=self.boundary_type_ymax

        if self.problem == 'blast':
            dico['radius'] = self.radius
            dico['center_x'] = self.center_x
            dico['center_y'] = self.center_y
            dico['density_in'] = self.density_in
            dico['density_out'] = self.density_out
            dico['pressure_in'] = self.pressure_in
            dico['pressure_out'] = self.pressure_out

        return dico
