import Hydro2d

import Hydro2d.ID
import Hydro2d.IP

myParamParse = Hydro2d.HydroParamParse("test_implode.ini")

# actually read file
Hydro2d.set_param_from_file(myParamParse)
Hydro2d.set_other_params(myParamParse)

par = Hydro2d.HydroParam(myParamParse)

qleft=zeros(Float64,Hydro2d.NBVAR)
qleft[ID]=1.0
qleft[IP]=1.0/(par.gamma0-1.0)
qright=zeros(Float64,Hydro2d.NBVAR)
qright[ID]=0.125
qright[IP]=0.14/(par.gamma0-1.0)

println("#########################################")
println("left  state",qleft)
println("right state",qright)
println()
try
    println("test riemann solver hllc")
    flux = Hydro2d.riemann_hllc(qleft, qright, par)
    println(flux)

    println("test riemann solver approx")
    flux = Hydro2d.riemann_approx(qleft, qright, par)
    println(flux)
catch e
    println(e)
end
println("#########################################")
