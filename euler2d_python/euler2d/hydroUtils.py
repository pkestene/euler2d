# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:38:54 2014

@author: pkestene
"""

import numpy as np
import hydroParam

ID    = hydroParam.ID
IP    = hydroParam.IP
IU    = hydroParam.IU
IV    = hydroParam.IV

NBVAR = hydroParam.NBVAR

FACE_XMIN = hydroParam.FACE_XMIN
FACE_XMAX = hydroParam.FACE_XMAX
FACE_YMIN = hydroParam.FACE_YMIN
FACE_YMAX = hydroParam.FACE_YMAX

IX    = hydroParam.IX
IY    = hydroParam.IY


def saturate(a):
        
    if np.isnan(a):
        a_sat=0

    if   a >= 1.0:
        a_sat = 1.0
        
    elif a <= 0:
        a_sat = 0.0

    else:
        a_sat = a
        
    return a_sat

######################################################
# `hydroUtils` class.
######################################################
class hydroUtils(object):
    
    #########################################
    def __init__(self, param):
        self.smallc     = param.smallc
        self.smallr     = param.smallr
        self.smallp     = param.smallp
        self.smallpp    = param.smallpp
        self.gamma0     = param.gamma0
        self.gamma6     = param.gamma6
        self.slope_type = param.slope_type

        # define a routine alias to the actual riemann solver routine
        riemann_solver_str = param.Config.get('HYDRO', 'riemannSolver','hllc')

        if   riemann_solver_str == 'hllc':

            self.riemann_2d = self.riemann_hllc

        elif riemann_solver_str == 'approx':

            self.riemann_2d = self.riemann_approx

        else: # default value

            self.riemann_2d = self.riemann_hllc

    #########################################
    def eos(self, rho, eint):
        
        """
        Equation of state:
        compute pressure p and speed of sound c, from density rho and
        internal energy eint (per mass unit) using the "calorically perfect gas"
        equation of state : 
        .. math::
            eint=\frac{p}{\rho (\gamma-1)}
            
        Recall that gamma is equal to the ratio of specific heats c_p/c_v.
        """
        
        p = max((self.gamma0 - 1.0) * rho * eint, rho * self.smallp)
        c = np.sqrt(self.gamma0 * p / rho)

        return p, c
        
    #########################################
    def computePrimitives(self, u):
        """
        Convert conservative variables (rho, rho*u, rho*v, e) to 
        primitive variables (rho,u,v,p)
        :param[in]  u:  conservative variables array
        :returns: q (primitive variables array) and c (local speed of sound)
        """

        q = np.array([0.0, 0.0, 0.0, 0.0])
        q[ID] = max(u[ID], self.smallr)
        q[IU] = u[IU] / q[ID]
        q[IV] = u[IV] / q[ID]
  
        eken = 0.5 * (q[IU] * q[IU] + q[IV] * q[IV])
        e    = u[IP] / q[ID] - eken
        if e < 0:
            print "FATAL ERROR : hydro eint < 0  : e % eken % d % u % v %".format(u[IP],eken,u[ID],u[IU],u[IV])
            return
  
        # compute pressure and speed of sound
        q[IP], c = self.eos(q[ID], e)

        return q, c
        
    #########################################
    def computePrimitives_ij(self, U, i, j):
        
        """
        Convert conservative variables (rho, rho*u, rho*v, e) to 
        primitive variables (rho,u,v,p)
        :param  U:  conservative variables (2D array)
        :param  i:  x-coordinate of current cell
        :param  j:    y-coordinate of current cell
        :returns:  qLoc (primitive variables of current cell) and 
        c (speed of sound)
        """
        uLoc = np.array([0.0, 0.0, 0.0, 0.0])

        # get primitive variables in current cell
        uLoc[ID] = U[i,j,ID]
        uLoc[IP] = U[i,j,IP]
        uLoc[IU] = U[i,j,IU]
        uLoc[IV] = U[i,j,IV]

        # return qLoc, c
        #return self.computePrimitives(uLoc)

        q = np.array([0.0, 0.0, 0.0, 0.0])
        q[ID] = max(uLoc[ID], self.smallr)
        q[IU] = uLoc[IU] / q[ID]
        q[IV] = uLoc[IV] / q[ID]
  
        eken = 0.5 * (q[IU] * q[IU] + q[IV] * q[IV])
        e    = uLoc[IP] / q[ID] - eken
        if e < 0:
            print "FATAL ERROR at {},{}: hydro eint < 0  : e {} eken {} d {} u {} v {}".format(i,j,uLoc[IP],eken,uLoc[ID],uLoc[IU],uLoc[IV])
            
            return
  
        # compute pressure and speed of sound
        q[IP], c = self.eos(q[ID], e)

        return q, c
        
    #########################################
    def cmpflx(self,qgdnv):
        """
        Convert Godunov state into flux.
        """
        
        flux = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Compute fluxes
        # Mass density
        flux[ID] = qgdnv[ID] * qgdnv[IU]
    
        # Normal momentum
        flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP]
      
        # Transverse momentum
        flux[IV] = flux[ID] * qgdnv[IV]
    
        # Total energy
        entho = 1.0 / (self.gamma0 - 1.0)
    
        ekin = 0.5 * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV])
      
        etot = qgdnv[IP] * entho + ekin
        flux[IP] = qgdnv[IU] * (etot + qgdnv[IP])
    
        return flux
        
    #########################################
    def  slope_unsplit_hydro_2d(self, q, qNeighbors):
        
        """
        Compute primitive variables slope (vector dq) from q and its neighbors.
        This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
        
        Only slope_type 1 and 2 are supported.
        
        :param  q           : current primitive variable state
        :param  qNeighbors  : states in the neighboring cells along XDIR and YDIR
        :returns: dqX,dqY : array of X and Y slopes
        """        

        qPlusX  = qNeighbors[0]
        qMinusX = qNeighbors[1]
        qPlusY  = qNeighbors[2]
        qMinusY = qNeighbors[3]

        dqX = np.zeros(NBVAR)
        dqY = np.zeros(NBVAR)

        if self.slope_type==1 or self.slope_type==2:  # minmod or average
    
            # slopes in first coordinate direction
            dlft = self.slope_type*(q      - qMinusX)
            drgt = self.slope_type*(qPlusX - q      )
            dcen = 0.5 *           (qPlusX - qMinusX)
            dsgn = np.sign(dcen)
            slop = np.minimum( abs(dlft), abs(drgt) )
            dlim = slop
            # mulitply by zero if negative
            dlim *= (np.sign(dlft*drgt)+1)/2
            dqX = dsgn * np.minimum( dlim, abs(dcen) )
      
            # slopes in second coordinate direction
            dlft = self.slope_type*(q      - qMinusY)
            drgt = self.slope_type*(qPlusY - q      )
            dcen = 0.5 *           (qPlusY - qMinusY)
            dsgn = np.sign(dcen)
            slop = np.minimum( abs(dlft), abs(drgt) )
            dlim = slop
            # mulitply by zero if negative
            dlim *= (np.sign(dlft*drgt)+1)/2
            dqY = dsgn * np.minimum( dlim, abs(dcen) )

        return dqX, dqY
        
    #########################################
    def trace_unsplit_2d(self, q, qNeighbors, c, dtdx, dtdy):
        
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

        # create returned lists
        qm = np.zeros((2,NBVAR))
        qp = np.zeros((2,NBVAR))

        # first compute slopes
        dqX, dqY = self.slope_unsplit_hydro_2d(q, qNeighbors)
          
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
        qp[IX,ID] = r - 0.5*drx + sr0*dtdx*0.5
        qp[IX,IP] = p - 0.5*dpx + sp0*dtdx*0.5
        qp[IX,IU] = u - 0.5*dux + su0*dtdx*0.5
        qp[IX,IV] = v - 0.5*dvx + sv0*dtdx*0.5
        qp[IX,ID] = max(self.smallr, qp[IX,ID])
          
        # Left state at right interface
        qm[IX,ID] = r + 0.5*drx + sr0*dtdx*0.5
        qm[IX,IP] = p + 0.5*dpx + sp0*dtdx*0.5
        qm[IX,IU] = u + 0.5*dux + su0*dtdx*0.5
        qm[IX,IV] = v + 0.5*dvx + sv0*dtdx*0.5
        qm[IX,ID] = max(self.smallr, qm[IX,ID])
          
        # Top state at bottom interface
        qp[IY,ID] = r - 0.5*dry + sr0*dtdy*0.5
        qp[IY,IP] = p - 0.5*dpy + sp0*dtdy*0.5
        qp[IY,IU] = u - 0.5*duy + su0*dtdy*0.5
        qp[IY,IV] = v - 0.5*dvy + sv0*dtdy*0.5
        qp[IY,ID] = max(self.smallr, qp[IY,ID])
          
        # Bottom state at top interface
        qm[IY,ID] = r + 0.5*dry + sr0*dtdy*0.5
        qm[IY,IP] = p + 0.5*dpy + sp0*dtdy*0.5
        qm[IY,IU] = u + 0.5*duy + su0*dtdy*0.5
        qm[IY,IV] = v + 0.5*dvy + sv0*dtdy*0.5
        qm[IY,ID] = max(self.smallr, qm[IY,ID])
  
        return qm, qp
        
    #########################################
    def trace_unsplit_hydro_2d(self, q, dq, dtdx, dtdy):

        # create returned lists
        qm = np.zeros((2,NBVAR))
        qp = np.zeros((2,NBVAR))
        
        # Cell centered values
        r = q[ID]
        p = q[IP]
        u = q[IU]
        v = q[IV]

        # Cell centered TVD slopes in X direction
        drx = dq[IX,ID];  drx *= 0.5
        dpx = dq[IX,IP];  dpx *= 0.5
        dux = dq[IX,IU];  dux *= 0.5
        dvx = dq[IX,IV];  dvx *= 0.5
  
        # Cell centered TVD slopes in Y direction
        dry = dq[IY,ID];  dry *= 0.5
        dpy = dq[IY,IP];  dpy *= 0.5
        duy = dq[IY,IU];  duy *= 0.5
        dvy = dq[IY,IV];  dvy *= 0.5

        # Source terms (including transverse derivatives)
        # only true for cartesian grid
        sr0 = (-u*drx-dux*r)            *dtdx + (-v*dry-dvy*r)            *dtdy
        su0 = (-u*dux-dpx/r)            *dtdx + (-v*duy      )            *dtdy
        sv0 = (-u*dvx      )            *dtdx + (-v*dvy-dpy/r)            *dtdy
        sp0 = (-u*dpx-dux*self.gamma0*p)*dtdx + (-v*dpy-dvy*self.gamma0*p)*dtdy    
        # end cartesian

        # Update in time the  primitive variables
        r = r + sr0
        u = u + su0
        v = v + sv0
        p = p + sp0

        # Face averaged right state at left interface
        qp[IX,ID] = r - drx
        qp[IX,IU] = u - dux
        qp[IX,IV] = v - dvx
        qp[IX,IP] = p - dpx
        qp[IX,ID] = max(self.smallr,  qp[IX,ID])
        qp[IX,IP] = max(self.smallp * qp[IX,ID], qp[IX,IP])
  
        # Face averaged left state at right interface
        qm[IX,ID] = r + drx
        qm[IX,IU] = u + dux
        qm[IX,IV] = v + dvx
        qm[IX,IP] = p + dpx
        qm[IX,ID] = max(self.smallr,  qm[IX,ID])
        qm[IX,IP] = max(self.smallp * qm[IX,ID], qm[IX,IP])

        # Face averaged top state at bottom interface
        qp[IY,ID] = r - dry
        qp[IY,IU] = u - duy
        qp[IY,IV] = v - dvy
        qp[IY,IP] = p - dpy
        qp[IY,ID] = max(self.smallr,  qp[IY,ID])
        qp[IY,IP] = max(self.smallp * qp[IY,ID], qp[IY,IP])
  
        # Face averaged bottom state at top interface
        qm[IY,ID] = r + dry
        qm[IY,IU] = u + duy
        qm[IY,IV] = v + dvy
        qm[IY,IP] = p + dpy
        qm[IY,ID] = max(self.smallr,  qm[IY,ID])
        qm[IY,IP] = max(self.smallp * qm[IY,ID], qm[IY,IP])

        return qm, qp

    #########################################
    def trace_unsplit_hydro_2d_by_direction(self, q, dqX, dqY, dtdx, dtdy, faceId):

        # create returned reconstructed state
        qface = np.zeros(NBVAR)
        
        # Cell centered values
        r = q[ID]
        p = q[IP]
        u = q[IU]
        v = q[IV]

        # Cell centered TVD slopes in X direction
        drx = dqX[ID];  drx *= 0.5
        dpx = dqX[IP];  dpx *= 0.5
        dux = dqX[IU];  dux *= 0.5
        dvx = dqX[IV];  dvx *= 0.5
  
        # Cell centered TVD slopes in Y direction
        dry = dqY[ID];  dry *= 0.5
        dpy = dqY[IP];  dpy *= 0.5
        duy = dqY[IU];  duy *= 0.5
        dvy = dqY[IV];  dvy *= 0.5

        # Source terms (including transverse derivatives)
        # only true for cartesian grid
        sr0 = (-u*drx-dux*r)            *dtdx + (-v*dry-dvy*r)            *dtdy
        su0 = (-u*dux-dpx/r)            *dtdx + (-v*duy      )            *dtdy
        sv0 = (-u*dvx      )            *dtdx + (-v*dvy-dpy/r)            *dtdy
        sp0 = (-u*dpx-dux*self.gamma0*p)*dtdx + (-v*dpy-dvy*self.gamma0*p)*dtdy    
        # end cartesian

        # Update in time the  primitive variables
        r = r + sr0
        u = u + su0
        v = v + sv0
        p = p + sp0

        if faceId == FACE_XMIN:
            # Face averaged right state at left interface
            qface[ID] = r - drx
            qface[IU] = u - dux
            qface[IV] = v - dvx
            qface[IP] = p - dpx
            qface[ID] = max(self.smallr,  qface[ID])
            qface[IP] = max(self.smallp * qface[ID], qface[IP])

        if faceId == FACE_XMAX:
            # Face averaged left state at right interface
            qface[ID] = r + drx
            qface[IU] = u + dux
            qface[IV] = v + dvx
            qface[IP] = p + dpx
            qface[ID] = max(self.smallr,  qface[ID])
            qface[IP] = max(self.smallp * qface[ID], qface[IP])

        if faceId == FACE_YMIN:
            # Face averaged top state at bottom interface
            qface[ID] = r - dry
            qface[IU] = u - duy
            qface[IV] = v - dvy
            qface[IP] = p - dpy
            qface[ID] = max(self.smallr,  qface[ID])
            qface[IP] = max(self.smallp * qface[ID], qface[IP])

        if faceId == FACE_YMAX:
            # Face averaged bottom state at top interface
            qface[ID] = r + dry
            qface[IU] = u + duy
            qface[IV] = v + dvy
            qface[IP] = p + dpy
            qface[ID] = max(self.smallr,  qface[ID])
            qface[IP] = max(self.smallp * qface[ID], qface[IP])

        return qface

    #########################################
    def riemann_approx(self, qleft, qright):
        """
        Riemann solver, equivalent to riemann_approx in RAMSES (see file
        godunov_utils.f90 in RAMSES).
        """
        
        qgdnv= np.zeros(NBVAR)
        flux = np.zeros(NBVAR)
    
        # Pressure, density and velocity
        rl = max(qleft [ID], self.smallr)
        ul =     qleft [IU]
        pl = max(qleft [IP], rl*self.smallp)
        rr = max(qright[ID], self.smallr)
        ur =     qright[IU]
        pr = max(qright[IP], rr*self.smallp)
  
        # Lagrangian sound speed
        cl = self.gamma0*pl*rl
        cr = self.gamma0*pr*rr
  
        # First guess
        wl = np.sqrt(cl)
        wr = np.sqrt(cr)
        pstar = max(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), 0.0)
        pold = pstar
        conv = 1.0

        # Newton-Raphson iterations to find pstar at the required accuracy
        iter=0
        niter_riemann=100
        while (iter < niter_riemann) and (conv > 1e-6):
    
          wwl = np.sqrt(cl*(1.0+self.gamma6*(pold-pl)/pl))
          wwr = np.sqrt(cr*(1.0+self.gamma6*(pold-pr)/pr))
          ql = 2.0*wwl*wwl*wwl/(wwl*wwl+cl)
          qr = 2.0*wwr*wwr*wwr/(wwr*wwr+cr)
          usl = ul-(pold-pl)/wwl
          usr = ur+(pold-pr)/wwr
          delp = max(qr*ql/(qr+ql)*(usl-usr),-pold)
          
          pold = pold+delp
          conv = abs(delp/(pold+self.smallpp))	 # Convergence indicator
          iter += 1
    
        # Star region pressure
        # for a two-shock Riemann problem
        pstar = pold
        wl = np.sqrt(cl*(1.0+self.gamma6*(pstar-pl)/pl))
        wr = np.sqrt(cr*(1.0+self.gamma6*(pstar-pr)/pr))
          
        # Star region velocity
        # for a two shock Riemann problem
        ustar = 0.5 * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr)
      
        # Left going or right going contact wave
        sgnm = np.copysign(1.0, ustar)
      
        # Left or right unperturbed state
      
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
    
        co = max(self.smallc, np.sqrt(abs(self.gamma0*po/ro)))
  
        # Star region density (Shock, max prevents vacuum formation in star region)
        rstar = max( (ro/(1.0+ro*(po-pstar)/(wo*wo))), self.smallr)
        # Star region sound speed
        cstar = max(self.smallc, np.sqrt(abs(self.gamma0*pstar/rstar)))
  
        # Compute rarefaction head and tail speed
        spout  = co    - sgnm*uo
        spin   = cstar - sgnm*ustar
        # Compute shock speed
        ushock = wo/ro - sgnm*uo
  
        if pstar >= po:
            spin  = ushock
            spout = ushock

        # Sample the solution at x/t=0
        scr = max(spout-spin, self.smallc+abs(spout+spin))
        frac = 0.5 * (1.0 + (spout + spin)/scr)
        #frac = SATURATE()
        if np.isnan(frac):
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
    
  
        return self.cmpflx(qgdnv)
        

    #########################################
    def riemann_hllc(self, qleft, qright):
        """
        Riemann solver HLLC.
        
        :param: qleft
        :param: qright
        :returns: qgdnv, flux
        """

        # returned flux init
        flux = np.zeros(NBVAR)
        
        # enthalpy
        entho = 1.0 / (self.gamma0 - 1.0)
        
        # Left variables
        rl = max(qleft[ID], self.smallr)
        pl = max(qleft[IP], rl*self.smallp)
        ul =     qleft[IU]
    
        ecinl  = 0.5*rl*ul*ul
        ecinl += 0.5*rl*qleft[IV]*qleft[IV]

        etotl = pl*entho+ecinl
        ptotl = pl

        # Right variables
        rr = max(qright[ID], self.smallr)
        pr = max(qright[IP], rr*self.smallp)
        ur =      qright[IU]

        ecinr = 0.5*rr*ur*ur
        ecinr += 0.5*rr*qright[IV]*qright[IV]
  
        etotr = pr*entho+ecinr
        ptotr = pr
    
        # Find the largest eigenvalues in the normal direction to the interface
        cfastl = np.sqrt(max(self.gamma0*pl/rl,self.smallc**2))
        cfastr = np.sqrt(max(self.gamma0*pr/rr,self.smallc**2))

        # Compute HLL wave speed
        SL = min(ul,ur) - max(cfastl,cfastr)
        SR = max(ul,ur) + max(cfastl,cfastr)

        # Compute lagrangian sound speed
        rcl = rl*(ul-SL)
        rcr = rr*(SR-ur)
    
        # Compute acoustic star state
        ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl)
        ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl)

        # Left star region variables
        rstarl    = rl*(SL-ul)/(SL-ustar)
        etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar)
    
        # Right star region variables
        rstarr    = rr*(SR-ur)/(SR-ustar)
        etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar)
    
        # Sample the solution at x/t=0
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
  
        return flux
        
