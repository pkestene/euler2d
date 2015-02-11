import ConfParser

conf = ConfParser.ConfParse("test_implode.ini")

ConfParser.parse_conf!(conf)

# cfl exists
cfl = parsefloat( ConfParser.retrieve(conf, "hydro", "cfl") )
@printf("cfl=%f\n",cfl)

# dummy doesn't exist
dummy=0.0
try
    dummy = parsefloat( ConfParser.retrieve(conf, "hydro", "dummy") )
catch
    dummy=1.0
end
@printf("dummy=%f\n",dummy)

# print current directory
println("current directory is ",splitdir(@__FILE__)[1])
