using Printf

mutable struct HydroRun

    # parameters
    par::HydroParam

    # for all implementation version
    U::Array{Float64,3}
    U2::Array{Float64,3}
    Q::Array{Float64,3}

    # specific to version 1
    # Qm_x::Array{Float64,3}
    # Qm_y::Array{Float64,3}
    # Qp_x::Array{Float64,3}
    # Qp_y::Array{Float64,3}

    HydroRun(par::HydroParam) = begin

        instance = new()

        instance.par = par

        isize = par.isize
        jsize = par.jsize

        # sort of memory allocation
        instance.U = zeros(Float64, isize,jsize,NBVAR)
        instance.U2 = zeros(Float64, isize,jsize,NBVAR)
        instance.Q = zeros(Float64, isize,jsize,NBVAR)

        if par.implementationVersion == 1
            # TODO
        end

        instance

    end

end # type HydroRun

####################################################################
####################################################################
function init_condition(hydroRun::HydroRun)

    par = hydroRun.par

    if par.problem == "implode"
        
        init_implode(hydroRun.U, par)
        init_implode(hydroRun.U2, par)

    elseif par.problem == "blast"

        init_blast(hydroRun.U, par)
        init_blast(hydroRun.U2, par)

    else

        printf("problem %s is not recognized")
        throw(DomainError())

    end

end # init_condition

####################################################################
####################################################################
function init_implode(U::Array{Float64,3}, 
                      par::HydroParam)

    gw = par.ghostWidth

    for i=1:par.isize
        for j=1:par.jsize
            tmp = 1.0*(i-gw)/par.nx + 1.0*(j-gw)/par.ny
            
            if tmp>0.5
                U[i,j,ID] = 1.0
                U[i,j,IP] = 1.0/(par.gamma0-1.0)
                U[i,j,IU] = 0.0
                U[i,j,IV] = 0.0
            else
                U[i,j,ID] = 0.125 #+0.1*rand(Uniform(0.0,1.0))
                U[i,j,IP] = 0.14/(par.gamma0-1.0)
                U[i,j,IU] = 0.0
                U[i,j,IV] = 0.0
            end
        end
    end

end # init_implode

####################################################################
####################################################################
@doc """
Compute time step satisfying CFL condition.
        
:param useU: specify which hydrodynamics data array
:returns: dt time step
""" ->
function compute_dt(hydroRun::HydroRun, 
                    useU::Int64)
    
    par = hydroRun.par

    if useU == 0
        U = hydroRun.U
    else
        U = hydroRun.U2
    end
        
    invDt = 0.0
    dx = par.dx
    dy = par.dy
    
    for i=1:par.isize
        for j=1:par.jsize
            qLoc, c = computePrimitives_ij(U,i,j, par)
            vx = c + abs(qLoc[IU])
            vy = c + abs(qLoc[IV])

            invDt = max(invDt, vx/dx + vy/dy)
            
        end
    end

    return par.cfl / invDt

end # compute_dt

####################################################################
####################################################################
@doc """
Convert conservative variables to primitive.
""" ->
function compute_primitives(U::Array{Float64,3}, 
                            Q::Array{Float64,3}, 
                            par::HydroParam)

    for i=1:par.isize
        for j=1:par.jsize
            qLoc, c = computePrimitives_ij(U,i,j,par)
            Q[i,j,:] = qLoc[:]
        end
    end
end # compute_primitives

####################################################################
####################################################################
@doc """
Fill ghost boundaries.
""" ->
function make_boundaries(U::Array{Float64,3}, par::HydroParam)

    gw     = par.ghostWidth
    b_xmin = par.boundary_type_xmin 
    b_xmax = par.boundary_type_xmax
    b_ymin = par.boundary_type_ymin 
    b_ymax = par.boundary_type_ymax 
    
    nx = par.nx
    ny = par.ny
    
    imin = par.imin
    imax = par.imax
    jmin = par.jmin
    jmax = par.jmax
    
    # boundary xmin
    for iVar=1:NBVAR
        for i=1:gw
            sign = 1.0
            if b_xmin == BC_DIRICHLET
                i0 = 2*gw+1-i
                if iVar==IU
                    sign = -1.0
                end
            elseif b_xmin == BC_NEUMANN
                i0 = gw+1
            else # periodic
                i0 = nx+i
            end
                
            for j=jmin+gw:jmax-gw
                U[i,j,iVar] = U[i0,j,iVar]*sign
            end
        end
    end

    # boundary xmax
    for iVar=1:NBVAR
        for i=nx+gw+1:nx+2*gw
            sign = 1.0
            if b_xmax == BC_DIRICHLET
                i0 = 2*nx + 2*gw+1-i
                if iVar==IU
                    sign = -1.0
                end
            elseif b_xmax == BC_NEUMANN
                i0 = nx+gw
            else  # periodic
                i0 = i-nx
            end
                
            for j=jmin+gw:jmax-gw
                U[i,j,iVar] = U[i0,j,iVar]*sign
            end
        end
    end
  
    # boundary ymin
    for iVar=1:NBVAR
        for j=1:gw
            sign = 1.0
            if b_ymin == BC_DIRICHLET
                j0 = 2*gw+1-j
                if iVar==IV
                    sign = -1.0
                end
            elseif b_ymin == BC_NEUMANN
                j0 = gw+1
            else  # periodic
                j0 = ny+j
            end
            
            for i=imin+gw:imax-gw
                U[i,j,iVar] =  U[i,j0,iVar]*sign
            end
        end
    end
        
    # boundary ymax
    for iVar=1:NBVAR
        for j=ny+gw+1:ny+2*gw
            sign = 1.0
            if b_ymax == BC_DIRICHLET
                j0 = 2*ny + 2*gw+1-j
                if iVar==IV
                    sign = -1.0
                end
            elseif b_ymax == BC_NEUMANN
                j0 = ny+gw
            else  # periodic
                j0 = j-ny
            end
            
            for i=imin+gw:imax-gw
                U[i,j,iVar] = U[i,j0,iVar]*sign
            end
        end
    end

end # make_boundaries

####################################################################
####################################################################
@doc """
Wrapper to main routine for performing one time step integration.
""" ->
function godunov_unsplit(hydroRun::HydroRun,nStep,dt)

    #godunov_unsplit_cpu(U , U2, dt, nStep)

end # godunov_unsplit

####################################################################
####################################################################
@doc """
This is the main routine for performing one time step integration.
""" ->
function godunov_unsplit_cpu(hydroRun::HydroRun, 
                             dt::Float64, 
                             nStep::Int64)
    
    par = hydroRun.par

    # choose riemann solver
    riemann_2d = riemann_hllc
    if par.riemannSolver == "approx"
        riemann_2d = riemann_approx
    end
    
    dtdx  = dt / par.dx
    dtdy  = dt / par.dy
    isize = par.isize
    jsize = par.jsize
    gw    = par.ghostWidth

    # just make aliases
    if nStep%2 == 0
        U  = hydroRun.U
        U2 = hydroRun.U2
    else
        U  = hydroRun.U2
        U2 = hydroRun.U
    end
    
    # fill ghost cell in data_in
    #hydroMonitoring.boundaries_timer.start()
    make_boundaries(U,par)
    #hydroMonitoring.boundaries_timer.stop()
    
    # copy U into U2
    U2[:,:,:] = U[:,:,:]
    
    # alias 
    Q = hydroRun.Q

    # main computation
    #hydroMonitoring.godunov_timer.start()

    # convert to primitive variables
    compute_primitives(U, Q, par)
    
    if par.implementationVersion==0
        
        for i=gw+1:isize-gw+1
            for j=gw+1:jsize-gw+1
	        
                # primitive variables in neighborhood
                qLoc = zeros(Float64,NBVAR)
                qLocN = zeros(Float64,NBVAR)
                qNeighbors = zeros(Float64,2*TWO_D,NBVAR)
                
                # get slopes in current cell
                qLoc[:] = Q[i,j,:] 
                
                qNeighbors[1,:] = Q[i+1,j  ,:]
                qNeighbors[2,:] = Q[i-1,j  ,:]
                qNeighbors[3,:] = Q[i  ,j+1,:]
                qNeighbors[4,:] = Q[i  ,j-1,:]
	        
                # compute slopes in current cell
                dqX, dqY = slope_unsplit_hydro_2d(qLoc, qNeighbors, par)
                
                ##################################
                # left interface along X direction
                ##################################
                
                # get primitive variables state vector in
                # left neighbor along X
                qLocN[:] = Q[i-1,j,:] 
                
                qNeighbors[1,:] = Q[i  ,j  ,:]
                qNeighbors[2,:] = Q[i-2,j  ,:]
                qNeighbors[3,:] = Q[i-1,j+1,:]
                qNeighbors[4,:] = Q[i-1,j-1,:]
                
                # compute slopes in left neighbor along X
                dqX_n, dqY_n = slope_unsplit_hydro_2d(qLocN, qNeighbors,par)
                
                #
                # Compute reconstructed states at left interface
                #  along X in current cell
                #
                
                # left interface : right state
                qright = trace_unsplit_hydro_2d_by_direction(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, par)
                
                # left interface : left state
                qleft = trace_unsplit_hydro_2d_by_direction(qLocN, dqX_n, dqY_n, dtdx, dtdy, FACE_XMAX,par)
                
                flux_x = riemann_2d(qleft,qright,par)
                
                ##################################
                # left interface along Y direction
                ##################################
                
                # get primitive variables state vector in
                # left neighbor along Y
                qLocN[:] = Q[i,j-1,:] 
                
                qNeighbors[1,:] = Q[i+1,j-1,:]
                qNeighbors[2,:] = Q[i-1,j-1,:]
                qNeighbors[3,:] = Q[i  ,j  ,:]
                qNeighbors[4,:] = Q[i  ,j-2,:]
	        
                # compute slopes in current cell
                dqX_n, dqY_n = slope_unsplit_hydro_2d(qLocN, qNeighbors,par)

                #
                # Compute reconstructed states at left interface
                #  along X in current cell
                #
                
                # left interface : right state
                qright = trace_unsplit_hydro_2d_by_direction(qLoc, dqX,   dqY, dtdx,  dtdy, FACE_YMIN,par)
                
                # left interface : left state
                qleft = trace_unsplit_hydro_2d_by_direction(qLocN, dqX_n, dqY_n,dtdx, dtdy, FACE_YMAX,par)
                
                qleft[IU], qleft[IV] = qleft[IV], qleft[IU] # watchout IU, IV permutation
	        
                qright[IU], qright[IV]  = qright[IV], qright[IU] # watchout IU, IV permutation
                
                flux_y = riemann_2d(qleft,qright,par)
                
                # swap flux_y components
                flux_y[IU], flux_y[IV] = flux_y[IV], flux_y[IU] 
                
                #
                # update hydro array
                #
                U2[i-1,j  ,:] += -flux_x[:].*dtdx
                U2[i  ,j  ,:] +=  flux_x[:].*dtdx
                
                U2[i  ,j-1,:] += -flux_y[:].*dtdy
                U2[i  ,j  ,:] +=  flux_y[:].*dtdy
            end
        end
        
    else
        # TODO
        println("Not implemented - TODO")
    end
    #hydroMonitoring.godunov_timer.stop()

end  # godunov_unsplit_cpu

####################################################################
####################################################################
function save_hdf5(U::Array{Float64,3}, 
                   filename::String)

    # save in HDF5 format
    current_dir = splitdir(@__FILE__)[1]

    full_filename = joinpath(current_dir, filename)

    f = HDF5.h5open(full_filename, "w")
    
    HDF5.write(f, "density", U[:,:,ID])
    HDF5.write(f, "energy", U[:,:,IP])
    HDF5.write(f, "vx", U[:,:,IU])
    HDF5.write(f, "vy", U[:,:,IV])

    HDF5.close(f)

end # save_hdf5


####################################################################
####################################################################
function write_xdmf(filename_prefix::String, 
                    totalNumberOfSteps::Int64, 
                    par::HydroParam)

    # concatene xdmf suffix
    xdmf_filename = filename_prefix * ".xmf"

    f=open(xdmf_filename, "w")

    # remove h5 suffix
    baseName = filename_prefix

    @printf(f,"<?xml version=\"1.0\" ?>\n")
    @printf(f,"<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
    @printf(f,"<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n")
    @printf(f,"  <Domain>\n")
    @printf(f,"    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n")
    
    # for each time step write a <grid> </grid> item
    startStep = 0
    stopStep  = totalNumberOfSteps
    deltaStep = par.nOutput

    for nStep=startStep:deltaStep:stopStep
        
        hdf5_filename = @sprintf("%s_%03d.h5",filename_prefix,nStep) 
        
        @printf(f,"    <Grid Name=\"%s\" GridType=\"Uniform\">\n",baseName )
        @printf(f,"    <Time Value=\"%d\" />\n", nStep                     )
        @printf(f,"      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"%d %d\"/>\n", par.isize, par.jsize )
    
        @printf(f,"    <Geometry Type=\"ORIGIN_DXDY\">\n"        )
        @printf(f,"    <DataStructure\n"                         )
        @printf(f,"       Name=\"Origin\"\n"                     )
        @printf(f,"       DataType=\"Double\"\n"                 )
        @printf(f,"       Dimensions=\"2\"\n"                    )
        @printf(f,"       Format=\"XML\">\n"                     )
        @printf(f,"       0 0\n"                                 )
        @printf(f,"    </DataStructure>\n"                       )
        @printf(f,"    <DataStructure\n"                         )
        @printf(f,"       Name=\"Spacing\"\n"                    )
        @printf(f,"       DataType=\"Double\"\n"                 )
        @printf(f,"       Dimensions=\"2\"\n"                    )
        @printf(f,"       Format=\"XML\">\n"                     )
        @printf(f,"       1 1\n"                                 )
        @printf(f,"    </DataStructure>\n"                       )
        @printf(f,"    </Geometry>\n"                            )
        
        # density
        @printf(f,"      <Attribute Center=\"Node\" Name=\"density\">" )
        @printf(f,"        <DataStructure"                             )
        @printf(f,"           DataType=\"Double\"\n"   )
        
        @printf(f,"           Dimensions=\"%d %d\"\n",par.isize,par.jsize)
        @printf(f,"           Format=\"HDF\">"                           )
        @printf(f,"           %s:/density\n",hdf5_filename               )
        @printf(f,"        </DataStructure>\n"                           )
        @printf(f,"      </Attribute>\n"                                 )
        @printf(f,"   </Grid>\n" )
    end

    # finalize grid file for the current time step
    @printf(f,"   </Grid>\n" )
    @printf(f," </Domain>\n" )
    @printf(f,"</Xdmf>\n"    )

    close(f)

end # write_xdmf
