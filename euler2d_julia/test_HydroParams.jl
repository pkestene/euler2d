# test features contained in HydroParams.jl

import Hydro2d

myParamParse = Hydro2d.HydroParamParse("test_implode.ini")


# cfl = parsefloat( ConfParser.retrieve(myParamParse.conf, "hydro", "cfl") )

# @printf("cfl=%f\n",cfl)

# cfl2 = Hydro2d.get_float(myParamParse, "hydro", "cfl2", 0.5)
# @printf("cfl2=%f\n",cfl2)

# nx = Hydro2d.get_int(myParamParse, "mesh", "nex", 15)
# @printf("nx=%d\n",nx)

Hydro2d.set_param_from_file(myParamParse)
Hydro2d.set_other_params(myParamParse)

println("############ myParamParse.dic #############")
println(myParamParse.dic)

println("############ myParamParse #############")
Hydro2d.printConfig(myParamParse)

myParam = Hydro2d.HydroParam(myParamParse)
println("############")
println(typeof(myParam))
println(myParam)


println("BC_PERIODIC = ",Hydro2d.BC_PERIODIC)
