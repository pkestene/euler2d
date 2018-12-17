import Hydro2d

using Printf

function main()

    # default parameter file
    filename="test_implode.ini"

    # 
    # parse command line
    #
    if length(ARGS) > 0
        # use first argument as parameter filename 
        # here it is UTF8String, will be converted into ASCIIString when passing
        # to ConfParser
        filename=ARGS[1]
    else
        println("Using default parameter file : ",filename)
    end
    
    # create parameter parser
    myParamParse = Hydro2d.HydroParamParse(filename)
    
    # actually read file
    Hydro2d.set_param_from_file(myParamParse)
    Hydro2d.set_other_params(myParamParse)
    
    par = Hydro2d.HydroParam(myParamParse)
    
    
    # print config
    Hydro2d.printConfig(myParamParse)


    #
    # Start simulation
    #

    t = 0.0
    dt = 0.0
    nStep = 0

    # create a hydroRun object
    hr = Hydro2d.HydroRun(par)
    
    # initial condition
    Hydro2d.init_condition(hr)

    # fill external boundaries
    Hydro2d.make_boundaries(hr.U, par)
    Hydro2d.make_boundaries(hr.U2, par)

    # create timers
    timer_total  = Hydro2d.Timer_t()
    timer_io     = Hydro2d.Timer_t()

    dt =  Hydro2d.compute_dt(hr,0)
    
    println("Start computation...")
    Hydro2d.timer_start(timer_total)
    #tic()

    # Hydrodynamics solver loop
    t     = 0.0
    nStep = 0
    #while t < par.tEnd && nStep < par.nStepmax
    while nStep < par.nStepmax
        # output
        Hydro2d.timer_start(timer_io)
        if nStep % par.nOutput == 0
            @printf("Output results at time t=%16.13f step %05d dt=%13.10f\n",t,nStep,dt)
            filename = @sprintf("%s_%03d.h5",par.filename_prefix,nStep)
            if nStep % 2 == 0
                Hydro2d.save_hdf5(hr.U,  filename)
            else
                Hydro2d.save_hdf5(hr.U2, filename)
            end

        end

        # write xdmf wrapper
        #xdmf_filename = @sprintf("U_%03d.xmf",nStep)
        Hydro2d.write_xdmf(par.filename_prefix,nStep,par)

        Hydro2d.timer_stop(timer_io)

        # compute new dt
        dt =  Hydro2d.compute_dt(hr,nStep%2)
    
        # perform one step integration
        Hydro2d.godunov_unsplit_cpu(hr, dt, nStep)

        # increase time
        nStep += 1
        t+=dt
    end
    #t_tot = toc()
    Hydro2d.timer_stop(timer_total)

    t_tot = Hydro2d.elapsed(timer_total)
    t_io  = Hydro2d.elapsed(timer_io)

    @printf("total       time : %5.3f secondes\n",t_tot)
    @printf("io          time : %5.3f secondes\n",t_io)
    @printf("Perf             : %10.2f number of cell-updates/s\n",nStep*par.isize*par.jsize/t_tot)

end # main


main()
