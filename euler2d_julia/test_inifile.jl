import IniFile
using Printf

ini = IniFile.Inifile()
# IniFile.read(ini, "test_implode.ini")

# cfl exists
cfl = IniFile.get( ini, "hydro", "cfl", 1.0)
@printf("cfl=%f\n",cfl)

# dummy doesn't exist
dummy=0.0
dummy = IniFile.get( ini, "hydro", "dummy", 1.0 )
@printf("dummy=%f\n",dummy)

# print current directory
println("current directory is ",splitdir(@__FILE__)[1])
