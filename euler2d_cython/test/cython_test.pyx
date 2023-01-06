# cython: language_level = 3

import numpy as np
cimport numpy as np

#from posix.time cimport timeval, timezone, gettimeofday

import hydroParam as hp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef int ID    = hp.ID
cdef int IP    = hp.IP
cdef int IU    = hp.IU
cdef int IV    = hp.IV

cdef int NBVAR = hp.NBVAR

cdef int toto
cdef int toto2

toto = 12
print("toto, toto2 = {} {}".format(toto,toto2))

toto2 = toto
toto2 -= 1
print("toto, toto2 = {} {}".format(toto,toto2))
print("id(toto), id(toto2) = {} {}".format(id(toto),id(toto2)))


par = hp.hydroParams('./test.ini')
