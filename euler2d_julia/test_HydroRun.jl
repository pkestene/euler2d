import Hydro2d

import Hydro2d.ID
import Hydro2d.IP
import Hydro2d.IU
import Hydro2d.IV

using HDF5

myParamParse = Hydro2d.HydroParamParse("test_implode.ini")

# actually read file
Hydro2d.set_param_from_file(myParamParse)
Hydro2d.set_other_params(myParamParse)

par = Hydro2d.HydroParam(myParamParse)

# create a hydroRun object
hydroRun = Hydro2d.HydroRun(par)

# initial condition
Hydro2d.init_condition(hydroRun)

# save in HDF5 format
current_dir = splitdir(@__FILE__)[1]

dt = Hydro2d.compute_dt(hydroRun,0)
println("dt=",dt)

filename = joinpath(current_dir, "test.h5")
f = h5open(filename, "w")

write(f, "density", float64(hydroRun.U[:,:,ID]))
write(f, "energy", float64(hydroRun.U[:,:,IP]))
write(f, "vx", float64(hydroRun.U[:,:,IU]))
write(f, "vy", float64(hydroRun.U[:,:,IV]))
close(f)

Hydro2d.save_hdf5(hydroRun.U, "test2.h5")
