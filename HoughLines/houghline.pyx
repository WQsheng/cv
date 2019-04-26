import numpy as np
import time

cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos


def timeit(method):
    def wraper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        print(method.__name__ + ": " + str(end-start))
        return result
    return wraper


@timeit
@cython.boundscheck(False)
@cython.wraparound(False)
def HoughLines(img, int rho, float theta):
    height, width = img.shape
    numangle = int(np.pi/theta)
    cdef int numrho
    numrho = int(((width + height) * 2 + 1) / rho)
    cdef float irho
    irho = 1 / rho

    accum = np.zeros((numangle + 2, numrho + 2), dtype=np.int32)
    tabSin = np.zeros((numangle, ), dtype=np.float32)
    tabCos = np.zeros((numangle, ), dtype=np.float32)
    cdef float[:] tabSin_ = tabSin
    cdef float[:] tabCos_ = tabCos
    cdef int [:, :] accum_ = accum

    cdef float ang=0
    cdef int n1=0
    for n1 in range(numangle):
        tabSin_[n1] = sin(ang * irho)
        tabCos_[n1] = cos(ang * irho)
        ang += theta

    # ###############################################################################################
    i_s, j_s = np.where(img > 0)

    cdef int i, j, n, r
    for i, j in zip(i_s, j_s):
        for n in range(numangle):
            r = int(j * tabCos_[n] + i * tabSin_[n])
            r += (numrho - 1) / 2
            accum_[n+1, r+1] += 1
    # ###############################################################################################

    ns, rs = np.where(accum > 0)
    lens = accum[ns, rs].reshape(-1, 1)
    ns = (ns - 1) * rho
    rs = (rs - (numrho-1)*0.5)*rho
    result = np.concatenate((rs.reshape(-1, 1), ns.reshape(-1, 1), lens), axis=1)

    return result
