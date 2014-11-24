# cython: infer_types = True
# cython: profile = False
# cython: boundscheck = False
# cython: wraparound = False
# cython: nonecheck = False
# cython: cdivision = True

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:38:54 2014

@author: pkestene
"""


import numpy as np
cimport numpy as np

cimport cython

import hydroParam

cdef int ID    = hydroParam.ID
cdef int IP    = hydroParam.IP
cdef int IU    = hydroParam.IU
cdef int IV    = hydroParam.IV

cdef int NBVAR = hydroParam.NBVAR

cdef int IX    = hydroParam.IX
cdef int IY    = hydroParam.IY

cdef int BC_DIRICHLET = hydroParam.BC_DIRICHLET
cdef int BC_NEUMANN   = hydroParam.BC_NEUMANN
cdef int BC_PERIODIC  = hydroParam.BC_PERIODIC
cdef int BC_COPY      = hydroParam.BC_COPY

# can also use enum if you prefer
# this is important to use anonymous enum here
# cdef enum:
#     ID    = 0
#     IP    = 1
#     IU    = 2
#     IV    = 3

#from libc.math cimport sqrt, fmin, fmax, sqrt, fabs

cdef extern from "math.h":
    bint isnan(double x)
    double fmin(double x, double y)
    double fmax(double x, double y)
    double sqrt(double x)
    double fabs(double x)
    double copysign(double x, double y)
    
cdef double saturate(double a):

    cdef double a_sat

    if isnan(a):
        a_sat=0

    if a >= 1.0:
        a_sat = 1.0
        
    elif a <= 0:
        a_sat = 0.0

    else:
        a_sat = a
        
    return a_sat

#ctypedef np.ndarray[double, ndim=1] (*riemann_fn_ptr)(np.ndarray[double, ndim=1],
#                                                     np.ndarray[double, ndim=1])

######################################################
# `hydroUtils` class.
######################################################
cdef class hydroUtils:

     #cdef public double smallc, smallr, smallp, smallpp, gamma0, gamma6
     #cdef public double slope_type

     #########################################
     def __init__(self, dict param):
         
         cdef str riemann_solver_str
         
         self.smallc     = param['smallc']
         self.smallr     = param['smallr']
         self.smallp     = param['smallp']
         self.smallpp    = param['smallpp']
         self.gamma0     = param['gamma0']
         self.gamma6     = param['gamma6']
         self.slope_type = param['slope_type']

         # define a routine alias to the actual riemann solver routine
         riemann_solver_str = param['riemannSolver']
         
         
#         if   riemann_solver_str == 'hllc':

#            self.riemann_2d = self.riemann_hllc

#         elif riemann_solver_str == 'approx':

#            self.riemann_2d = self.riemann_approx

#         else: # default value

#            self.riemann_2d = self.riemann_hllc


     #########################################
     cdef eos(self, double rho, double eint, double *p, double *c):
        
         """
         Equation of state:
         compute pressure p and speed of sound c, from density rho and
         internal energy eint (per mass unit) using the "calorically perfect gas"
         equation of state : 
         .. math::
             eint=\frac{p}{\rho (\gamma-1)}
            
         Recall that gamma is equal to the ratio of specific heats c_p/c_v.
         """
         #cdef double p, c
         #cdef EOS_out output
         
         p[0] = fmax((self.gamma0 - 1.0) * rho * eint, rho * self.smallp)
         c[0] = sqrt(self.gamma0 * p[0] / rho)
     
     #########################################
     cdef computePrimitives(self, double* u, double* q, double *c):
         """
         Convert conservative variables (rho, rho*u, rho*v, e) to 
         primitive variables (rho,u,v,p)
         :param[in]  u:  conservative variables array
         :param[out] q:  primitive variables array
         :param[out] c:  speed of sound
         """
         cdef double eken, e
         cdef int ID =0
         
         q[ID] = fmax(u[ID], self.smallr)
         q[IU] = u[IU] / q[ID]
         q[IV] = u[IV] / q[ID]
 
         eken = 0.5 * (q[IU] * q[IU] + q[IV] * q[IV])
         e    = u[IP] / q[ID] - eken
         if e < 0:
             print "FATAL ERROR : hydro eint < 0  : e % eken % d % u % v %".format(u[IP],eken,u[ID],u[IU],u[IV])
 
         # compute pressure and speed of sound
         #cdef EOS_out output
         self.eos(q[ID], e, &(q[IP]), c)

     #########################################
     cpdef double computePrimitives_ij(self,
                                       np.ndarray[double, ndim=3] U,
                                       int i,
                                       int j,
                                       np.ndarray[double, ndim=1] qLoc):
        
         """
         Convert conservative variables (rho, rho*u, rho*v, e) to 
         primitive variables (rho,u,v,p)
         :param  U:  conservative variables (2D array)
         :param  i:  x-coordinate of current cell
         :param  j:    y-coordinate of current cell
         :returns:  qLoc (primitive variables of current cell) and 
         c (speed of sound)
         """
         cdef np.ndarray[double, ndim=1] uLoc = np.zeros(NBVAR, dtype=np.double)

         cdef double eken, e, c
         
         # get primitive variables in current cell
         uLoc[ID] = U[i,j,ID]
         uLoc[IP] = U[i,j,IP]
         uLoc[IU] = U[i,j,IU]
         uLoc[IV] = U[i,j,IV]

         # return c
         #return self.computePrimitives(uLoc)

         qLoc[ID] = fmax(uLoc[ID], self.smallr)
         qLoc[IU] = uLoc[IU] / qLoc[ID]
         qLoc[IV] = uLoc[IV] / qLoc[ID]
  
         eken = 0.5 * (qLoc[IU] * qLoc[IU] + qLoc[IV] * qLoc[IV])
         e    = uLoc[IP] / qLoc[ID] - eken
         if e < 0:
             print "FATAL ERROR at {},{}: hydro eint < 0  : e {} eken {} d {} u {} v {}".format(i,j,uLoc[IP],eken,uLoc[ID],uLoc[IU],uLoc[IV])
            
         # compute pressure and speed of sound
         #qLoc[IP], c = self.eos(qLoc[ID], e)
         #cdef EOS_out output = self.eos(qLoc[ID], e)
         #cdef EOS_out output
         self.eos(qLoc[ID], e, &(qLoc[IP]), &c)
         
         return c
        
     #########################################
     cdef cmpflx(self, double[4] qgdnv, double[4] flux):
         """
         Convert Godunov state into flux.
         :param: qgdnv (input)
         :param: flux (output)
         """
        
         # Compute fluxes
         # Mass density
         flux[ID] = qgdnv[ID] * qgdnv[IU]
    
         # Normal momentum
         flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP]
      
         # Transverse momentum
         flux[IV] = flux[ID] * qgdnv[IV]
    
         # Total energy
         cdef double entho = 1.0 / (self.gamma0 - 1.0)
    
         cdef double ekin = 0.5 * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV])
      
         cdef double etot = qgdnv[IP] * entho + ekin
         flux[IP] = qgdnv[IU] * (etot + qgdnv[IP])
    
        
     #########################################
     cdef slope_unsplit_hydro_2d(self,
                                 double[4] q,
                                 double[4][4] qNeighbors,
                                 double[4] dqX,
                                 double[4] dqY):
        
         """
         Compute primitive variables slope (vector dq) from q and its neighbors.
         This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
        
         Only slope_type 1 and 2 are supported.
        
         :param  q           : current primitive variable state
         :param  qNeighbors  : states in the neighboring cells along XDIR and YDIR
         #:returns: dqX,dqY : array of X and Y slopes
         """        

         cdef double* qPlusX  = qNeighbors[0]
         cdef double* qMinusX = qNeighbors[1]
         cdef double* qPlusY  = qNeighbors[2]
         cdef double* qMinusY = qNeighbors[3]

         cdef double[4] dlft
         cdef double[4] drgt
         cdef double[4] dcen
         cdef double[4] dsgn
         cdef double[4] slop
         cdef double[4] dlim
         
         cdef int ivar
         
         if self.slope_type==1 or self.slope_type==2:  # minmod or average
    
            for ivar in range(NBVAR):
             
                # slopes in first coordinate direction
                dlft[ivar] = self.slope_type*(q[ivar]      - qMinusX[ivar])
                drgt[ivar] = self.slope_type*(qPlusX[ivar] - q[ivar]      )
                dcen[ivar] = 0.5 *           (qPlusX[ivar] - qMinusX[ivar])
                dsgn[ivar] = copysign(1.0,dcen[ivar])
                slop[ivar] = fmin( fabs(dlft[ivar]), fabs(drgt[ivar]) )
                dlim[ivar] = slop[ivar]
                # multiply by zero if negative
                if dlft[ivar]*drgt[ivar] <= 0.0:
                    dlim[ivar] = 0.0
                dqX[ivar] = dsgn[ivar] * fmin( dlim[ivar], fabs(dcen[ivar]) )
      
                # slopes in second coordinate direction
                dlft[ivar] = self.slope_type*(q[ivar]      - qMinusY[ivar])
                drgt[ivar] = self.slope_type*(qPlusY[ivar] - q[ivar]      )
                dcen[ivar] = 0.5 *           (qPlusY[ivar] - qMinusY[ivar])
                dsgn[ivar] = copysign(1.0,dcen[ivar])
                slop[ivar] = fmin( fabs(dlft[ivar]), fabs(drgt[ivar]) )
                dlim[ivar] = slop[ivar]
                # multiply by zero if negative
                if dlft[ivar]*drgt[ivar] <= 0.0:
                    dlim[ivar] = 0.0
                dqY[ivar] = dsgn[ivar] * fmin( dlim[ivar], fabs(dcen[ivar]) )

     #########################################
     cdef trace_unsplit_2d(self,
                           double[4] q,
                           double[4][4] qNeighbors,
                           double c,
                           double dtdx,
                           double dtdy,
                           double[2][4] qm,
                           double[2][4] qp):
       
        """
        Trace computations for unsplit Godunov scheme.
 
        :param q          : Primitive variables state.
        :param qNeighbors : state in the neighbor cells (2 neighbors
        per dimension, in the following order x+, x-, y+, y-, z+, z-)
        :param c:         : local sound speed.
        :param dtdx:      : dt over dx
        :param dtdy:      : dt over dy
        : returns:        : qm and qp states (one per dimension)
        """

        # first compute slopes
        cdef double[4] dqX
        cdef double[4] dqY
        
        cdef double r, p, u, v
        cdef double drx, dpx, dux, dvx
        cdef double dry, dpy, duy, dvy
        cdef double sr0, sp0, su0, sv0
                
        #dqX, dqY = self.slope_unsplit_hydro_2d(q, qNeighbors)
        self.slope_unsplit_hydro_2d(q, qNeighbors, dqX, dqY)
          
        # Cell centered values
        r =  q[ID]
        p =  q[IP]
        u =  q[IU]
        v =  q[IV]
          
        # TVD slopes in all directions
        drx = dqX[ID]
        dpx = dqX[IP]
        dux = dqX[IU]
        dvx = dqX[IV]
          
        dry = dqY[ID]
        dpy = dqY[IP]
        duy = dqY[IU]
        dvy = dqY[IV]
          
        # source terms (with transverse derivatives)
        sr0 = -u*drx-v*dry - (dux+dvy)*r
        sp0 = -u*dpx-v*dpy - (dux+dvy)*self.gamma0*p
        su0 = -u*dux-v*duy - (dpx    )/r
        sv0 = -u*dvx-v*dvy - (dpy    )/r
          
        # Right state at left interface
        qp[IX][ID] = r - 0.5*drx + sr0*dtdx*0.5
        qp[IX][IP] = p - 0.5*dpx + sp0*dtdx*0.5
        qp[IX][IU] = u - 0.5*dux + su0*dtdx*0.5
        qp[IX][IV] = v - 0.5*dvx + sv0*dtdx*0.5
        qp[IX][ID] = fmax(self.smallr, qp[IX][ID])
          
        # Left state at right interface
        qm[IX][ID] = r + 0.5*drx + sr0*dtdx*0.5
        qm[IX][IP] = p + 0.5*dpx + sp0*dtdx*0.5
        qm[IX][IU] = u + 0.5*dux + su0*dtdx*0.5
        qm[IX][IV] = v + 0.5*dvx + sv0*dtdx*0.5
        qm[IX][ID] = fmax(self.smallr, qm[IX][ID])
          
        # Top state at bottom interface
        qp[IY][ID] = r - 0.5*dry + sr0*dtdy*0.5
        qp[IY][IP] = p - 0.5*dpy + sp0*dtdy*0.5
        qp[IY][IU] = u - 0.5*duy + su0*dtdy*0.5
        qp[IY][IV] = v - 0.5*dvy + sv0*dtdy*0.5
        qp[IY][ID] = fmax(self.smallr, qp[IY][ID])
          
        # Bottom state at top interface
        qm[IY][ID] = r + 0.5*dry + sr0*dtdy*0.5
        qm[IY][IP] = p + 0.5*dpy + sp0*dtdy*0.5
        qm[IY][IU] = u + 0.5*duy + su0*dtdy*0.5
        qm[IY][IV] = v + 0.5*dvy + sv0*dtdy*0.5
        qm[IY][ID] = fmax(self.smallr, qm[IY][ID])
  

     #########################################
     cdef trace_unsplit_hydro_2d(self,
                                 double[4] q,
                                 double[4] dqX,
                                 double[4] dqY,
                                 double dtdx,
                                 double dtdy,
                                 double[2][4] qm,
                                 double[2][4] qp):
                          

        #qm[:] = 0.0
        #qp[:] = 0.0
        
        # Cell centered values
        cdef double r = q[ID]
        cdef double p = q[IP]
        cdef double u = q[IU]
        cdef double v = q[IV]

        # Cell centered TVD slopes in X direction
        cdef double drx = dqX[ID] * 0.5
        cdef double dpx = dqX[IP] * 0.5
        cdef double dux = dqX[IU] * 0.5
        cdef double dvx = dqX[IV] * 0.5
  
        # Cell centered TVD slopes in Y direction
        cdef double dry = dqY[ID] * 0.5
        cdef double dpy = dqY[IP] * 0.5
        cdef double duy = dqY[IU] * 0.5
        cdef double dvy = dqY[IV] * 0.5

        # Source terms (including transverse derivatives)
        # only true for cartesian grid
        cdef double sr0 = (-u*drx-dux*r)            *dtdx + (-v*dry-dvy*r)            *dtdy
        cdef double su0 = (-u*dux-dpx/r)            *dtdx + (-v*duy      )            *dtdy
        cdef double sv0 = (-u*dvx      )            *dtdx + (-v*dvy-dpy/r)            *dtdy
        cdef double sp0 = (-u*dpx-dux*self.gamma0*p)*dtdx + (-v*dpy-dvy*self.gamma0*p)*dtdy    
        # end cartesian

        # Update in time the  primitive variables
        r = r + sr0
        u = u + su0
        v = v + sv0
        p = p + sp0

        # Face averaged right state at left interface
        qp[IX][ID] = r - drx
        qp[IX][IU] = u - dux
        qp[IX][IV] = v - dvx
        qp[IX][IP] = p - dpx
        qp[IX][ID] = fmax(self.smallr,  qp[IX][ID])
        qp[IX][IP] = fmax(self.smallp * qp[IX][ID], qp[IX][IP])
  
        # Face averaged left state at right interface
        qm[IX][ID] = r + drx
        qm[IX][IU] = u + dux
        qm[IX][IV] = v + dvx
        qm[IX][IP] = p + dpx
        qm[IX][ID] = fmax(self.smallr,  qm[IX][ID])
        qm[IX][IP] = fmax(self.smallp * qm[IX][ID], qm[IX][IP])

        # Face averaged top state at bottom interface
        qp[IY][ID] = r - dry
        qp[IY][IU] = u - duy
        qp[IY][IV] = v - dvy
        qp[IY][IP] = p - dpy
        qp[IY][ID] = fmax(self.smallr,  qp[IY][ID])
        qp[IY][IP] = fmax(self.smallp * qp[IY][ID], qp[IY][IP])
  
        # Face averaged bottom state at top interface
        qm[IY][ID] = r + dry
        qm[IY][IU] = u + duy
        qm[IY][IV] = v + dvy
        qm[IY][IP] = p + dpy
        qm[IY][ID] = fmax(self.smallr,  qm[IY][ID])
        qm[IY][IP] = fmax(self.smallp * qm[IY][ID], qm[IY][IP])

     #########################################
     cdef riemann_2d(self,
                     double* qleft, 
                     double* qright,
                     double* flux):
                          
        self.riemann_approx(qleft, qright, flux)                                      
                          

     #########################################
     cdef riemann_approx(self, 
                         double* qleft, 
                         double* qright,
                         double* flux):
         """
         Riemann solver, equivalent to riemann_approx in RAMSES (see file
         godunov_utils.f90 in RAMSES).
         """
        
         cdef double[4] qgdnv
   
         # Pressure, density and velocity
         cdef double rl = fmax(qleft [ID], self.smallr)
         cdef double ul =      qleft [IU]
         cdef double pl = fmax(qleft [IP], rl*self.smallp)
         cdef double rr = fmax(qright[ID], self.smallr)
         cdef double ur =      qright[IU]
         cdef double pr = fmax(qright[IP], rr*self.smallp)
  
         # Lagrangian sound speed
         cdef double cl = self.gamma0*pl*rl
         cdef double cr = self.gamma0*pr*rr
  
         # First guess
         cdef double wl = sqrt(cl)
         cdef double wr = sqrt(cr)
         cdef double pstar = fmax(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), 0.0)
         cdef double pold = pstar
         cdef double conv = 1.0

         # Newton-Raphson iterations to find pstar at the required accuracy
         cdef int iter=0
         cdef int niter_riemann=100
         cdef double wwl, wwr, ql, qr, usl, usr, delp
         while (iter < niter_riemann) and (conv > 1e-6):
    
           wwl = sqrt(cl*(1.0+self.gamma6*(pold-pl)/pl))
           wwr = sqrt(cr*(1.0+self.gamma6*(pold-pr)/pr))
           ql = 2.0*wwl*wwl*wwl/(wwl*wwl+cl)
           qr = 2.0*wwr*wwr*wwr/(wwr*wwr+cr)
           usl = ul-(pold-pl)/wwl
           usr = ur+(pold-pr)/wwr
           delp = fmax(qr*ql/(qr+ql)*(usl-usr),-pold)
          
           pold = pold+delp
           conv = fabs(delp/(pold+self.smallpp))	 # Convergence indicator
           iter += 1
    
         # Star region pressure
         # for a two-shock Riemann problem
         pstar = pold
         wl = sqrt(cl*(1.0+self.gamma6*(pstar-pl)/pl))
         wr = sqrt(cr*(1.0+self.gamma6*(pstar-pr)/pr))
          
         # Star region velocity
         # for a two shock Riemann problem
         ustar = 0.5 * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr)
      
         # Left going or right going contact wave
         sgnm = copysign(1.0, ustar)
      
         # Left or right unperturbed state
         cdef double ro, uo, po, wo
         if sgnm > 0.0:
             ro = rl
             uo = ul
             po = pl
             wo = wl
         else:
             ro = rr
             uo = ur
             po = pr
             wo = wr
    
         co = fmax(self.smallc, sqrt(fabs(self.gamma0*po/ro)))
  
         # Star region density (Shock, max prevents vacuum formation in star region)
         rstar = fmax( (ro/(1.0+ro*(po-pstar)/(wo*wo))), self.smallr)
         # Star region sound speed
         cstar = fmax(self.smallc, sqrt(fabs(self.gamma0*pstar/rstar)))
  
         # Compute rarefaction head and tail speed
         spout  = co    - sgnm*uo
         spin   = cstar - sgnm*ustar
         # Compute shock speed
         ushock = wo/ro - sgnm*uo
  
         if pstar >= po:
             spin  = ushock
             spout = ushock

         # Sample the solution at x/t=0
         scr = fmax(spout-spin, self.smallc+fabs(spout+spin))
         frac = 0.5 * (1.0 + (spout + spin)/scr)
         #frac = SATURATE()
         if isnan(frac):
             frac = 0.0
         else:
             frac = saturate(frac)
  
         qgdnv[ID] = frac*rstar + (1.0-frac)*ro
         qgdnv[IU] = frac*ustar + (1.0-frac)*uo
         qgdnv[IP] = frac*pstar + (1.0-frac)*po
  
         if spout < 0.0:
             qgdnv[ID] = ro
             qgdnv[IU] = uo
             qgdnv[IP] = po
  
         if spin > 0.0:
             qgdnv[ID] = rstar
             qgdnv[IU] = ustar
             qgdnv[IP] = pstar
  
         # transverse velocity
         if sgnm > 0.0:
             qgdnv[IV] = qleft[IV]
         else:
             qgdnv[IV] = qright[IV]
    
  
         self.cmpflx(qgdnv, flux)
        
     #########################################
     cdef riemann_hllc(self, 
                       double* qleft, 
                       double* qright,
                       double* flux):
         """
         Riemann solver HLLC.
        
         :param: qleft (input)
         :param: qright (input)
         :param: flux (output)
         """

         # enthalpy
         cdef double entho = 1.0 / (self.gamma0 - 1.0)
        
         # Left variables
         cdef double rl = fmax(qleft[ID], self.smallr)
         cdef double pl = fmax(qleft[IP], rl*self.smallp)
         cdef double ul =      qleft[IU]
    
         cdef double ecinl  = 0.5*rl*ul*ul
         ecinl += 0.5*rl*qleft[IV]*qleft[IV]

         cdef double etotl = pl*entho+ecinl
         cdef double ptotl = pl

         # Right variables
         cdef double rr = fmax(qright[ID], self.smallr)
         cdef double pr = fmax(qright[IP], rr*self.smallp)
         cdef double ur =      qright[IU]

         cdef double ecinr =  0.5*rr*ur*ur
         ecinr += 0.5*rr*qright[IV]*qright[IV]
  
         cdef double etotr = pr*entho+ecinr
         cdef double ptotr = pr
    
         # Find the largest eigenvalues in the normal direction to the interface
         cdef double cfastl = sqrt(fmax(self.gamma0*pl/rl,self.smallc**2))
         cdef double cfastr = sqrt(fmax(self.gamma0*pr/rr,self.smallc**2))

         # Compute HLL wave speed
         cdef double SL = fmin(ul,ur) - fmax(cfastl,cfastr)
         cdef double SR = fmax(ul,ur) + fmax(cfastl,cfastr)

         # Compute lagrangian sound speed
         cdef double rcl = rl*(ul-SL)
         cdef double rcr = rr*(SR-ur)
    
         # Compute acoustic star state
         cdef double ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl)
         cdef double ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl)

         # Left star region variables
         cdef double rstarl    = rl*(SL-ul)/(SL-ustar)
         cdef double etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar)
    
         # Right star region variables
         cdef double rstarr    = rr*(SR-ur)/(SR-ustar)
         cdef double etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar)
    
         # Sample the solution at x/t=0
         cdef double ro, uo, ptoto, etoto         
         if SL > 0.0:
             ro=rl
             uo=ul
             ptoto=ptotl
             etoto=etotl
         elif ustar > 0.0:
             ro=rstarl
             uo=ustar
             ptoto=ptotstar
             etoto=etotstarl
         elif SR > 0.0:
             ro=rstarr
             uo=ustar
             ptoto=ptotstar
             etoto=etotstarr
         else:
             ro=rr
             uo=ur
             ptoto=ptotr
             etoto=etotr
      
         # Compute the Godunov flux
         flux[ID] = ro*uo
         flux[IU] = ro*uo*uo+ptoto
         flux[IP] = (etoto+ptoto)*uo
         if flux[ID] > 0.0:
             flux[IV] = flux[ID]*qleft[IV]
         else:
             flux[IV] = flux[ID]*qright[IV]
  
