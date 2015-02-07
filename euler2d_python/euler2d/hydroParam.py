# -*- coding: utf-8 -*-
"""
Define class hydroParams.

"""

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

        import ConfigParser
        
        # parser input parameter file
        self.Config = ConfigParser.ConfigParser()
        self.Config.read(iniFilename)

    ########################################################
    def _setParamFromFile(self):
        """
        Parse ini file, and set parameters.
        """

        self.tEnd     = self.Config.getfloat('RUN','tEnd')
        self.nStepmax = self.Config.getint('RUN','nStepmax')
        self.nOutput  = self.Config.getint('RUN','nOutput')

        self.nx       = self.Config.getint('MESH','nx')
        self.ny       = self.Config.getint('MESH','ny')

        self.boundary_type_xmin = self.Config.getint('MESH','boundary_type_xmin')
        self.boundary_type_xmax = self.Config.getint('MESH','boundary_type_xmax')
        self.boundary_type_ymin = self.Config.getint('MESH','boundary_type_ymin')
        self.boundary_type_ymax = self.Config.getint('MESH','boundary_type_ymax')
    
        self.gamma0        = self.Config.getfloat('HYDRO','gamma0')
        self.cfl           = self.Config.getfloat('HYDRO','cfl')
        self.niter_riemann = self.Config.getint('HYDRO','niter_riemann') 
        self.iorder        = self.Config.getint('HYDRO','iorder') 
        self.slope_type    = self.Config.getint('HYDRO','slope_type') 
        self.problem       = self.Config.get('HYDRO','problem','implode') 

        # parse problem specific parameters
        if self.problem == 'blast':

            if self.Config.has_section('blast'):

                if self.Config.has_option('blast','radius'):
                    self.radius = self.Config.getfloat('blast','radius')
                else:
                    self.radius = 1.0

                if self.Config.has_option('blast','center_x'):
                    self.center_x = self.Config.getfloat('blast','center_x')
                else:
                    self.center_x = self.nx/2.0


                if self.Config.has_option('blast','center_y'):
                    self.center_y = self.Config.getfloat('blast','center_y')
                else:
                    self.center_y = self.ny/2.0


                if self.Config.has_option('blast','density_in'):
                    self.density_in = self.Config.getfloat('blast','density_in')
                else:
                    self.density_in = 1.0


                if self.Config.has_option('blast','density_out'):
                    self.density_out = self.Config.getfloat('blast','density_out')
                else:
                    self.density_out = 1.0


                if self.Config.has_option('blast','pressure_in'):
                    self.pressure_in = self.Config.getfloat('blast','pressure_in')
                else:
                    self.pressure_in = 10.0


                if self.Config.has_option('blast','pressure_out'):
                    self.pressure_out = self.Config.getfloat('blast','pressure_out')
                else:
                    self.pressure_out = 0.1

                
            else:
                print('Error: param file should have section named \'blast\'')
        
        self.implementationVersion  = self.Config.getint('OTHER','implementationVersion') 
                
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

        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
  
        self.smallc  = 1e-7
        self.smallr  = 1e-7
        self.smallp  = self.smallc * self.smallc / self.gamma0
        self.smallpp = self.smallr * self.smallp
        self.gamma6  = (self.gamma0 + 1.0)/(2.0 * self.gamma0)


    ########################################################
    def printConfig(self):

        print "##########################"
        print "Simulation run parameters:"
        print "##########################"
        print "nx         : {}".format(self.nx)
        print "ny         : {}".format(self.ny)
        print "dx         : {}".format(self.dx)
        print "dy         : {}".format(self.dy)
        print "imin       : {}".format(self.imin)
        print "imax       : {}".format(self.imax)
        print "jmin       : {}".format(self.jmin)
        print "jmax       : {}".format(self.jmax)
        print "nStepmax   : {}".format(self.nStepmax)
        print "tEnd       : {}".format(self.tEnd)
        print "nOutput    : {}".format(self.nOutput)
        print "gamma0     : {}".format(self.gamma0)
        print "cfl        : {}".format(self.cfl)
        print "smallr     : {}".format(self.smallr)
        print "smallc     : {}".format(self.smallc)
        print "iorder     : {}".format(self.iorder)
        print "slope_type : {}".format(self.slope_type)
        print "problem    : {}".format(self.problem)

