using ConfParser

const ID=1  # ID Density field index
const IP=2  # IP Pressure/Energy field index
const IU=3  # X velocity / momentum index
const IV=4  # Y velocity / momentum index

const NBVAR = 4

const TWO_D = 2

# identifying direction
const IX=1
const IY=2

# identifying directions for reconstructed state on cell faces
const FACE_XMIN=1
const FACE_XMAX=2
const FACE_YMIN=3
const FACE_YMAX=4

# type of boundary condition (note that BC_COPY is only used in the
# MPI version for inside boundary)
# enum BoundaryConditionType 
const BC_UNDEFINED = 1
const BC_DIRICHLET = 2   # reflecting border condition
const BC_NEUMANN   = 3   # absorbing border condition
const BC_PERIODIC  = 4   # periodic border condition
const BC_COPY      = 5   # only used in MPI parallelized version

#
# type HydroParamParse
#
type HydroParamParse
    iniFilename::String
    conf::ConfParse
    dic::Dict
    
    # usefull Constructor
    HydroParamParse(filename::String) = begin
        instance = new()

        instance.iniFilename = filename
        try
            # take care that ConfParse only accept ASCIIString !!!
            instance.conf = ConfParse(ASCIIString(filename))
            ConfParser.parse_conf!(instance.conf)
        catch e
            println("Failed to parsed ini file : $e")
        end
        instance.dic = Dict{String,Any}()
        instance
    end
end # type HydroParamParse

function get_int(paramParse::HydroParamParse,section::String,name::String, default=0)
    
    returned = default
    try
        returned = parseint( ConfParser.retrieve(paramParse.conf, section, name) )
    catch e
        println("get_int failed to parse requested value :",section," ",name)
        println("$e")
    end
    return returned
end # get_int

function get_float(paramParse::HydroParamParse,section::String,name::String, default=0.0)
    
    returned = default
    try
        returned = parsefloat( ConfParser.retrieve(paramParse.conf, section, name) )
    catch e
        println("get_float failed to parse requested value :",section," ",name)
        println("$e")
    end
    return returned
end # get_float

function get_bool(paramParse::HydroParamParse,section::String,name::String, default::Bool=false)
    
    returned = default
    try
        data = ConfParser.retrieve(paramParse.conf, section, name)
        if data != "yes" && data != "true" && data != "True"
            returned = false
        else
            returned = true
        end
    end
    return returned
end # get_bool

function get_string(paramParse::HydroParamParse,section::String,name::String, default::String="")
    
    returned = default
    try
        returned = ConfParser.retrieve(paramParse.conf, section, name)
    end
    return returned
end # get_string

#
# type HydroParam
#
type HydroParam

    # from parameter file
    tEnd::Float64
    nStepmax::Int32
    nOutput::Int32
    
    nx::Int32
    ny::Int32

    boundary_type_xmin::Int32
    boundary_type_xmax::Int32
    boundary_type_ymin::Int32
    boundary_type_ymax::Int32

    gamma0::Float64
    cfl::Float64
    niter_riemann::Int32
    iorder::Int32
    slope_type::Int32
    problem::String
    riemannSolver::String
    implementationVersion::Int32

    filename_prefix::String

    # other
    ghostWidth::Int32

    imin::Int32
    imax::Int32
    jmin::Int32
    jmax::Int32
    isize::Int32
    jsize::Int32

    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64

    dx::Float64
    dy::Float64

    smallc::Float64
    smallr::Float64
    smallp::Float64
    smallpp::Float64
    gamma6::Float64

    # constructor from a HydroParamParse object
    HydroParam(paramParse::HydroParamParse) = begin
        par = new()

        # from file
        par.tEnd = paramParse.dic["tEnd"]
        par.nStepmax = paramParse.dic["nStepmax"]
        par.nOutput = paramParse.dic["nOutput"]
        
        par.nx = paramParse.dic["nx"]
        par.ny = paramParse.dic["ny"]
        
        par.boundary_type_xmin = paramParse.dic["boundary_type_xmin"]
        par.boundary_type_xmax = paramParse.dic["boundary_type_xmax"]
        par.boundary_type_ymin = paramParse.dic["boundary_type_ymin"]
        par.boundary_type_ymax = paramParse.dic["boundary_type_ymax"]
        
        par.gamma0 = paramParse.dic["gamma0"]
        par.cfl = paramParse.dic["cfl"]
        par.niter_riemann = paramParse.dic["niter_riemann"]
        par.iorder = paramParse.dic["iorder"]
        par.slope_type = paramParse.dic["slope_type"]
        par.problem = paramParse.dic["problem"]
        par.riemannSolver = paramParse.dic["riemannSolver"]
        par.implementationVersion = paramParse.dic["implementationVersion"]
        par.filename_prefix = paramParse.dic["filename_prefix"]

        # other
        par.ghostWidth = 2
        par.imin = paramParse.dic["imin"]
        par.jmin = paramParse.dic["jmin"]
        
        par.imax = paramParse.dic["imax"]
        par.jmax = paramParse.dic["jmax"]
        
        par.isize = paramParse.dic["isize"]
        par.jsize = paramParse.dic["jsize"]
        
        par.xmin = paramParse.dic["xmin"]
        par.xmax = paramParse.dic["xmax"]
        par.ymin = paramParse.dic["ymin"]
        par.ymax = paramParse.dic["ymax"]
        
        par.dx = paramParse.dic["dx"]
        par.dy = paramParse.dic["dy"]
        
        par.smallc = paramParse.dic["smallc"]
        par.smallr = paramParse.dic["smallr"]
        par.smallp = paramParse.dic["smallp"]
        par.smallpp = paramParse.dic["smallpp"]
        par.gamma6 = paramParse.dic["gamma6"]

        par
    end

end # type HydroParam

function set_param_from_file(paramParse::HydroParamParse)

    paramParse.dic["tEnd"] = get_float(paramParse,"run","tEnd", 0.0)
    paramParse.dic["nStepmax"] = get_int(paramParse,"run","nStepmax")
    paramParse.dic["nOutput"] = get_int(paramParse,"run","nOutput")
    
    paramParse.dic["nx"]       = get_int(paramParse,"mesh","nx")
    paramParse.dic["ny"]       = get_int(paramParse,"mesh","ny")
    
    paramParse.dic["boundary_type_xmin"] = get_int(paramParse,"mesh","boundary_type_xmin")
    paramParse.dic["boundary_type_xmax"] = get_int(paramParse,"mesh","boundary_type_xmax")
    paramParse.dic["boundary_type_ymin"] = get_int(paramParse,"mesh","boundary_type_ymin")
    paramParse.dic["boundary_type_ymax"] = get_int(paramParse,"mesh","boundary_type_ymax")
    
    paramParse.dic["gamma0"]        = get_float(paramParse,"hydro","gamma0")
    paramParse.dic["cfl"]           = get_float(paramParse,"hydro","cfl")
    paramParse.dic["niter_riemann"] = get_int(paramParse,"hydro","niter_riemann") 
    paramParse.dic["iorder"]        = get_int(paramParse,"hydro","iorder") 
    paramParse.dic["slope_type"]    = get_int(paramParse,"hydro","slope_type") 
    paramParse.dic["problem"]       = get_string(paramParse,"hydro","problem","implode") 
    paramParse.dic["riemannSolver"] = get_string(paramParse,"hydro", "riemannSolver", "hllc")
    paramParse.dic["implementationVersion"] = get_int(paramParse,"other", "implementationVersion",0)
    paramParse.dic["filename_prefix"] = get_string(paramParse,"other", "filename_prefix", "U")

end # set_param_from_file

function set_other_params(paramParse::HydroParamParse)

    dic = paramParse.dic

    dic["ghostWidth"] = 2

    dic["imin"] = 1
    dic["jmin"] = 1

    dic["imax"] = dic["nx"] + 2*dic["ghostWidth"]
    dic["jmax"] = dic["ny"] + 2*dic["ghostWidth"]
    
    dic["isize"] = dic["imax"] - dic["imin"] + 1
    dic["jsize"] = dic["jmax"] - dic["jmin"] + 1
    
    dic["xmin"] = 0.0
    dic["xmax"] = 1.0
    dic["ymin"] = 0.0
    dic["ymax"] = 1.0
    
    dic["dx"] = (dic["xmax"] - dic["xmin"]) / dic["nx"]
    dic["dy"] = (dic["ymax"] - dic["ymin"]) / dic["ny"]
    
    dic["smallc"]  = 1e-7
    dic["smallr"]  = 1e-7
    dic["smallp"]  = dic["smallc"] * dic["smallc"] / dic["gamma0"]
    dic["smallpp"] = dic["smallr"] * dic["smallp"]
    dic["gamma6"]  = (dic["gamma0"] + 1.0)/(2.0 * dic["gamma0"])
    
end

function printConfig(paramParse::HydroParamParse)

    println("##########################")
    println("Simulation run parameters:")
    println("##########################")

    for i=paramParse.dic
        println(i[1],"=",i[2])
    end

end # printConfig

