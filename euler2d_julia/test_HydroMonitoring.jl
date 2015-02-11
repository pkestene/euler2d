import Hydro2d

t = Hydro2d.Timer_t()


Hydro2d.timer_start(t)

println("Tes Ouf   ???")
println("Tes Ouf 2 ???")

Hydro2d.timer_stop(t)


println(Hydro2d.elapsed(t))
