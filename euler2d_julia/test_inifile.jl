import IniFile
using Printf

ini = IniFile.Inifile()
IniFile.read(ini, "test_implode.ini")

# cfl exists
cfl = IniFile.get_float(ini, "HYDRO", "cfl")
@printf("cfl=%f\n",cfl)

# dummy doesn't exist
local dummy
dummy=2.0
try
    dummy = IniFile.get_float(ini, "HYDRO", "dummy")
catch e
    println("item \"dummy\" not found; use the default value")
end
@printf("dummy=%f\n",dummy)

# print current directory
println("current directory is ",splitdir(@__FILE__)[1])
