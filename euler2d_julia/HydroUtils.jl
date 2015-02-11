####################################################################
####################################################################
function eos(rho::Float64,
             eint::Float64,
             par::HydroParam)

    p = max((par.gamma0 - 1.0) * rho * eint, rho * par.smallp)
    c = sqrt(par.gamma0 * p / rho)
    
    return p, c

end # eos

####################################################################
####################################################################
function saturate(a::Float64)
    
    a_sat = a

    if isnan(a)
        a_sat=0.0
    end
    
    if  a >= 1.0
        a_sat = 1.0        
    elseif a <= 0.0
        a_sat = 0.0
    else
        a_sat = a
    end
    
    return a_sat
end # saturate


#riemann_solver_dic=""

####################################################################
####################################################################
@doc """
Convert conservative variables (rho, rho*u, rho*v, e) to 
primitive variables (rho,u,v,p)
:param  U:  conservative variables (2D array)
:param  i:  x-coordinate of current cell
:param  j:    y-coordinate of current cell
:returns:  qLoc (primitive variables of current cell) and 
c (speed of sound)
""" ->
function computePrimitives_ij(U::Array{Float64,3},
                              i::Int64, 
                              j::Int64,
                              par::HydroParam)
        
    uLoc = zeros(Float64, NBVAR)
    
    # get primitive variables in current cell
    uLoc[ID] = U[i,j,ID]
    uLoc[IP] = U[i,j,IP]
    uLoc[IU] = U[i,j,IU]
    uLoc[IV] = U[i,j,IV]
    

    q = zeros(Float64, NBVAR)
    q[ID] = max(uLoc[ID], par.smallr)
    q[IU] = uLoc[IU] / q[ID]
    q[IV] = uLoc[IV] / q[ID]
    
    eken = 0.5 * (q[IU] * q[IU] + q[IV] * q[IV])
    e    = uLoc[IP] / q[ID] - eken
    if e < 0
        printf("FATAL ERROR at %d,%d: hydro eint < 0  : e %f eken %f d %f u %f v %f",i,j,uLoc[IP],eken,uLoc[ID],uLoc[IU],uLoc[IV])
        throw(DomainError())
    end
    
    # compute pressure and speed of sound
    q[IP], c = eos(q[ID], e, par)
    
    return q, c

end # computePrimitives_ij    

####################################################################
####################################################################
@doc """
Convert Godunov state into flux.
""" ->
function cmpflx(qgdnv::Array{Float64,1},
                par::HydroParam)
        
    flux = zeros(Float64, NBVAR)
    
    # Compute fluxes
    # Mass density
    flux[ID] = qgdnv[ID] * qgdnv[IU]
    
    # Normal momentum
    flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP]
    
    # Transverse momentum
    flux[IV] = flux[ID] * qgdnv[IV]
    
    # Total energy
    entho = 1.0 / (par.gamma0 - 1.0)
    
    ekin = 0.5 * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV])
    
    etot = qgdnv[IP] * entho + ekin
    flux[IP] = qgdnv[IU] * (etot + qgdnv[IP])
    
    return flux
end

####################################################################
####################################################################
@doc """
# Riemann solver HLLC.

# :param: qleft
# :param: qright
# :returns: qgdnv, flux
# """ ->
function riemann_hllc(qleft::Array{Float64,1}, 
                      qright::Array{Float64,1}, 
                      par::HydroParam)

    # returned flux init
    flux = zeros(Float64,NBVAR)
    
    # enthalpy
    entho = 1.0 / (par.gamma0 - 1.0)
    
    # Left variables
    rl = max(qleft[ID], par.smallr)
    pl = max(qleft[IP], rl*par.smallp)
    ul =     qleft[IU]
    
    ecinl  = 0.5*rl*ul*ul
    ecinl += 0.5*rl*qleft[IV]*qleft[IV]
    
    etotl = pl*entho+ecinl
    ptotl = pl
    
    # Right variables
    rr = max(qright[ID], par.smallr)
    pr = max(qright[IP], rr*par.smallp)
    ur =      qright[IU]
    
    ecinr = 0.5*rr*ur*ur
    ecinr += 0.5*rr*qright[IV]*qright[IV]
    
    etotr = pr*entho+ecinr
    ptotr = pr
    
    # Find the largest eigenvalues in the normal direction to the interface
    cfastl = sqrt(max(par.gamma0*pl/rl,par.smallc^2))
    cfastr = sqrt(max(par.gamma0*pr/rr,par.smallc^2))
    
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
    if SL > 0.0
        ro=rl
        uo=ul
        ptoto=ptotl
        etoto=etotl
    elseif ustar > 0.0
        ro=rstarl
        uo=ustar
        ptoto=ptotstar
        etoto=etotstarl
    elseif SR > 0.0
        ro=rstarr
        uo=ustar
        ptoto=ptotstar
        etoto=etotstarr
    else
        ro=rr
        uo=ur
        ptoto=ptotr
        etoto=etotr
    end
    
    # Compute the Godunov flux
    flux[ID] = ro*uo
    flux[IU] = ro*uo*uo+ptoto
    flux[IP] = (etoto+ptoto)*uo
    if flux[ID] > 0.0
        flux[IV] = flux[ID]*qleft[IV]
    else
        flux[IV] = flux[ID]*qright[IV]
    end
    
    return flux
end

####################################################################
####################################################################
@doc """
Riemann solver, equivalent to riemann_approx in RAMSES (see file
godunov_utils.f90 in RAMSES).
""" ->
function riemann_approx(qleft::Array{Float64,1}, 
                        qright::Array{Float64,1}, 
                        par::HydroParam)
        
    qgdnv= zeros(Float64,NBVAR)
    flux = zeros(Float64,NBVAR)
    
    # Pressure, density and velocity
    rl = max(qleft [ID], par.smallr)
    ul =     qleft [IU]
    pl = max(qleft [IP], rl*par.smallp)
    rr = max(qright[ID], par.smallr)
    ur =     qright[IU]
    pr = max(qright[IP], rr*par.smallp)
    
    # Lagrangian sound speed
    cl = par.gamma0*pl*rl
    cr = par.gamma0*pr*rr
    
    # First guess
    wl = sqrt(cl)
    wr = sqrt(cr)
    pstar = max(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), 0.0)
    pold = pstar
    conv = 1.0
    
    # Newton-Raphson iterations to find pstar at the required accuracy
    iter=0
    niter_riemann=100
    while (iter < niter_riemann) && (conv > 1e-6)
        
        wwl = sqrt(cl*(1.0+par.gamma6*(pold-pl)/pl))
        wwr = sqrt(cr*(1.0+par.gamma6*(pold-pr)/pr))
        ql = 2.0*wwl*wwl*wwl/(wwl*wwl+cl)
        qr = 2.0*wwr*wwr*wwr/(wwr*wwr+cr)
        usl = ul-(pold-pl)/wwl
        usr = ur+(pold-pr)/wwr
        delp = max(qr*ql/(qr+ql)*(usl-usr),-pold)
        
        pold = pold+delp
        conv = abs(delp/(pold+par.smallpp))	 # Convergence indicator
        iter += 1
    end
    
    # Star region pressure
    # for a two-shock Riemann problem
    pstar = pold
    wl = sqrt(cl*(1.0+par.gamma6*(pstar-pl)/pl))
    wr = sqrt(cr*(1.0+par.gamma6*(pstar-pr)/pr))
    
    # Star region velocity
    # for a two shock Riemann problem
    ustar = 0.5 * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr)
    
    # Left going or right going contact wave
    sgnm = copysign(1.0, ustar)
    
    # Left or right unperturbed state
    
    if sgnm > 0.0
        ro = rl
        uo = ul
        po = pl
        wo = wl
    else
        ro = rr
        uo = ur
        po = pr
        wo = wr
    end
       
    co = max(par.smallc, sqrt(abs(par.gamma0*po/ro)))
    
    # Star region density (Shock, max prevents vacuum formation in star region)
    rstar = max( (ro/(1.0+ro*(po-pstar)/(wo*wo))), par.smallr)
    # Star region sound speed
    cstar = max(par.smallc, sqrt(abs(par.gamma0*pstar/rstar)))
    
    # Compute rarefaction head and tail speed
    spout  = co    - sgnm*uo
    spin   = cstar - sgnm*ustar
    # Compute shock speed
    ushock = wo/ro - sgnm*uo
    
    if pstar >= po
        spin  = ushock
        spout = ushock
    end
    
    # Sample the solution at x/t=0
    scr = max(spout-spin, par.smallc+abs(spout+spin))
    frac = 0.5 * (1.0 + (spout + spin)/scr)
    #frac = SATURATE()
    if isnan(frac)
        frac = 0.0
    else
        frac = saturate(frac)
    end
  
    qgdnv[ID] = frac*rstar + (1.0-frac)*ro
    qgdnv[IU] = frac*ustar + (1.0-frac)*uo
    qgdnv[IP] = frac*pstar + (1.0-frac)*po
    
    if spout < 0.0
        qgdnv[ID] = ro
        qgdnv[IU] = uo
        qgdnv[IP] = po
    end
    
    if spin > 0.0
        qgdnv[ID] = rstar
        qgdnv[IU] = ustar
        qgdnv[IP] = pstar
    end

    # transverse velocity
    if sgnm > 0.0
        qgdnv[IV] = qleft[IV]
    else
        qgdnv[IV] = qright[IV]
    end
     
    return cmpflx(qgdnv,par)

end # riemann_approx

####################################################################
####################################################################
@doc """
Compute primitive variables slope (vector dq) from q and its neighbors.
This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.

Only slope_type 1 and 2 are supported.

:param  q           : current primitive variable state
:param  qNeighbors  : states in the neighboring cells along XDIR and YDIR
:returns: dqX,dqY : array of X and Y slopes
""" ->       
function slope_unsplit_hydro_2d(q::Array{Float64,1}, 
                                qNeighbors::Array{Float64,2}, 
                                par::HydroParam)
    
    # make sure qPlusX and other have the same shape as q
    qPlusX  = reshape(qNeighbors[1,:],NBVAR)
    qMinusX = reshape(qNeighbors[2,:],NBVAR)
    qPlusY  = reshape(qNeighbors[3,:],NBVAR)
    qMinusY = reshape(qNeighbors[4,:],NBVAR)
    
    dqX = zeros(Float64, NBVAR)
    dqY = zeros(Float64, NBVAR)

    if par.slope_type==1 || par.slope_type==2  # minmod or average
        
        # slopes in first coordinate direction
        dlft = par.slope_type*(q      - qMinusX)
        drgt = par.slope_type*(qPlusX - q      )
        dcen = 0.5 *          (qPlusX - qMinusX)
        dsgn = sign(dcen)
        slop = min( abs(dlft), abs(drgt) )
        dlim = slop
        # mulitply by zero if negative
        dlim .*= (sign(dlft .* drgt)+1)/2
        dqX = dsgn .* min( dlim, abs(dcen) )
        
        # slopes in second coordinate direction
        dlft = par.slope_type*(q      - qMinusY)
        drgt = par.slope_type*(qPlusY - q      )
        dcen = 0.5 *          (qPlusY - qMinusY)
        dsgn = sign(dcen)
        slop = min( abs(dlft), abs(drgt) )
        dlim = slop
        # mulitply by zero if negative
        dlim .*= (sign(dlft .* drgt)+1)/2
        dqY = dsgn .* min( dlim, abs(dcen) )
    end
    
    return dqX, dqY

end # slope_unsplit_hydro_2d

####################################################################
####################################################################
function trace_unsplit_hydro_2d_by_direction(q::Array{Float64,1}, 
                                             dqX::Array{Float64,1},
                                             dqY::Array{Float64,1},
                                             dtdx::Float64, 
                                             dtdy::Float64,
                                             faceId::Int64,
                                             par::HydroParam)

    # create returned reconstructed state
    qface = zeros(Float64,NBVAR)
        
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
    sr0 = (-u*drx-dux*r)           *dtdx + (-v*dry-dvy*r)           *dtdy
    su0 = (-u*dux-dpx/r)           *dtdx + (-v*duy      )           *dtdy
    sv0 = (-u*dvx      )           *dtdx + (-v*dvy-dpy/r)           *dtdy
    sp0 = (-u*dpx-dux*par.gamma0*p)*dtdx + (-v*dpy-dvy*par.gamma0*p)*dtdy    
    # end cartesian

    # Update in time the  primitive variables
    r = r + sr0
    u = u + su0
    v = v + sv0
    p = p + sp0
    
    if faceId == FACE_XMIN
        # Face averaged right state at left interface
        qface[ID] = r - drx
        qface[IU] = u - dux
        qface[IV] = v - dvx
        qface[IP] = p - dpx
        qface[ID] = max(par.smallr,  qface[ID])
        qface[IP] = max(par.smallp * qface[ID], qface[IP])
    end

    if faceId == FACE_XMAX
        # Face averaged left state at right interface
        qface[ID] = r + drx
        qface[IU] = u + dux
        qface[IV] = v + dvx
        qface[IP] = p + dpx
        qface[ID] = max(par.smallr,  qface[ID])
        qface[IP] = max(par.smallp * qface[ID], qface[IP])
    end
    
    if faceId == FACE_YMIN
        # Face averaged top state at bottom interface
        qface[ID] = r - dry
        qface[IU] = u - duy
        qface[IV] = v - dvy
        qface[IP] = p - dpy
        qface[ID] = max(par.smallr,  qface[ID])
        qface[IP] = max(par.smallp * qface[ID], qface[IP])
    end
       
    if faceId == FACE_YMAX
        # Face averaged bottom state at top interface
        qface[ID] = r + dry
        qface[IU] = u + duy
        qface[IV] = v + dvy
        qface[IP] = p + dpy
        qface[ID] = max(par.smallr,  qface[ID])
        qface[IP] = max(par.smallp * qface[ID], qface[IP])
    end

    return qface

end # trace_unsplit_hydro_2d_by_direction
