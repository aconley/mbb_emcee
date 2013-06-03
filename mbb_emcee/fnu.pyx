""" Cython version of the f_nu evaluation for a modified blackbody"""

import numpy as np
import cython
cimport numpy as np
from libc.math cimport expm1

def fnueval_thin_noalpha(np.ndarray[np.float64_t, ndim=1] freq, double T, 
                         double beta, double normfac):
    """ Evaluate fnu for optically thin, no alpha model"""

    assert freq.dtype == np.float64
    cdef np.float64_t h = 6.6260693e-34 #J/s
    cdef np.float64_t k = 1.3806505e-23 #J/K
    cdef np.float64_t hokt9 = 1e9 * h / (k * T)

    cdef int nfreq = freq.shape[0]
    cdef np.float64_t bp3 = beta + 3.0
    cdef np.ndarray[np.float64_t, ndim=1] retval =\
        np.zeros(nfreq, dtype=np.float64)

    for i in range(nfreq):
        cx = hokt9 * freq[i]
        retval[i] = normfac * cx**bp3 / expm1(cx)

    return retval

@cython.boundscheck(False)
def fnueval_thin_walpha(np.ndarray[np.float64_t, ndim=1] freq, double T, 
                        double beta, double alpha, double normfac, 
                        double xmerge, double kappa):
    """ Evaluate fnu for optically thin, with alpha model"""
	
    assert freq.dtype == np.float64
    cdef np.float64_t h = 6.6260693e-34 #J/s
    cdef np.float64_t k = 1.3806505e-23 #J/K
    cdef np.float64_t hokt9 = 1e9 * h / (k * T)

    cdef int nfreq = freq.shape[0]
    cdef np.float64_t bp3 = beta + 3.0
    cdef np.ndarray[np.float64_t, ndim=1] retval =\
        np.zeros(nfreq, dtype=np.float64)

    for i in range(nfreq):
        cx = hokt9 * freq[i]
        if cx > xmerge:
            retval[i] = kappa * cx**(-alpha)
        else:
            retval[i] = cx**bp3 / expm1(cx)

    return normfac * retval


@cython.boundscheck(False)
def fnueval_thick_noalpha(np.ndarray[np.float64_t, ndim=1] freq, double T, 
                          double beta, double x0, double normfac):
    """ Evaluate fnu for optically thick, no alpha model"""	

    assert freq.dtype == np.float64
    cdef np.float64_t h = 6.6260693e-34 #J/s
    cdef np.float64_t k = 1.3806505e-23 #J/K
    cdef np.float64_t hokt9 = 1e9 * h / (k * T)

    cdef int nfreq = freq.shape[0]
    cdef int i
    cdef np.float64_t cx
    cdef np.float64_t x0b
    cdef np.ndarray[np.float64_t, ndim=1] retval =\
        np.zeros(nfreq, dtype=np.float64)
 
    for i in range(nfreq):
        cx = hokt9 * freq[i]
        x0b = (cx / x0)**beta
        retval[i] = - normfac * expm1(-x0b) * cx**3 / expm1(cx)

    return retval

@cython.boundscheck(False)
def fnueval_thick_walpha(np.ndarray[np.float64_t, ndim=1] freq, double T, 
                         double beta, double x0, double alpha, double normfac, 
                         double xmerge, double kappa):
    """ Evaluate fnu for optically thick, with alpha model"""

    assert freq.dtype == np.float64

    cdef np.float64_t h = 6.6260693e-34 #J/s
    cdef np.float64_t k = 1.3806505e-23 #J/K
    cdef np.float64_t hokt9 = 1e9 * h / (k * T)

    cdef int nfreq = freq.shape[0]
    cdef int i
    cdef np.float64_t cx
    cdef np.float64_t x0b
    cdef np.ndarray[np.float64_t, ndim=1] retval =\
        np.zeros(nfreq, dtype=np.float64)
    

    for i in range(nfreq):
        cx = hokt9 * freq[i]
        if cx > xmerge:
            retval[i] = kappa * cx**(-alpha)
        else:
            x0b = (cx / x0)**beta
            retval[i] = - expm1(-x0b) * cx**3 / expm1(cx)

    return normfac * retval
