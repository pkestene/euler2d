# cython: language_level = 3

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  10 10:00:54 2014

@author: pkestene
"""

import numpy as np
cimport numpy as np

cdef struct EOS_out:
    double p
    double c

################################################################################
# `hydroUtils` class.
################################################################################
cdef class hydroUtils:

    cdef public double smallc, smallr, smallp, smallpp, gamma0, gamma6
    cdef public double slope_type

    cdef eos(self, double rho, double eint, double* p, double *c)

    cdef computePrimitives(self, double* u, double* q, double* c)

    cpdef double computePrimitives_ij(self,
                                      np.ndarray[double, ndim=3] U,
                                      int i,
                                      int j,
                                      np.ndarray[double, ndim=1] qLoc)

    cdef cmpflx(self, double[4] qgdnv, double[4] flux)

    cdef slope_unsplit_hydro_2d(self,
                                double[4] q,
                                double[4][4] qNeighbors,
                                double[4] dqX,
                                double[4] dqY)

    cdef trace_unsplit_2d(self,
                          double[4] q,
                          double[4][4] qNeighbors,
                          double c,
                          double dtdx,
                          double dtdy,
                          double[2][4] qm,
                          double[2][4] qp)

    cdef trace_unsplit_hydro_2d(self,
                                double[4] q,
                                double[4] dqX,
                                double[4] dqY,
                                double dtdx,
                                double dtdy,
                                double[2][4] qm,
                                double[2][4] qp)

    cdef trace_unsplit_hydro_2d_by_direction(self,
                                             double[4] q,
                                             double[4] dqX,
                                             double[4] dqY,
                                             double dtdx,
                                             double dtdy,
                                             int faceId,
                                             double[4] qface)

    cdef riemann_2d(self,
                    double[4] qleft,
                    double[4] qright,
                    double[4] flux)

    cdef riemann_approx(self,
                        double[4] qleft,
                        double[4] qright,
                        double[4] flux)

    cdef riemann_hllc(self,
                      double[4] qleft,
                      double[4] qright,
                      double[4] flux)
