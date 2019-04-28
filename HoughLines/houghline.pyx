import numpy as np
import time
import math
from cython.parallel import prange, parallel, threadid

cimport cython
from libc.math cimport sin, cos


def timeit(method):
    def wraper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        print(method.__name__ + ": " + str(end-start))
        return result
    return wraper


@cython.boundscheck(False)
@cython.wraparound(False)
def HoughLines(img, int rho, float theta):
    cdef int height, width, numangle, numrho
    cdef float irho
    height, width = img.shape
    numangle = int(math.pi/theta)
    numrho = int(((width + height) * 2 + 1) / rho)
    irho = 1 / rho

    accum = np.zeros((numangle + 2, numrho + 2), dtype=np.int32)
    tabSin = np.zeros((numangle, ), dtype=np.float32)
    tabCos = np.zeros((numangle, ), dtype=np.float32)
    cdef float[:] tabSin_ = tabSin
    cdef float[:] tabCos_ = tabCos
    cdef int [:, :] accum_ = accum
    cdef unsigned char[:, :] img_ = img

    cdef float ang=0
    cdef int n1=0

    for n1 in range(numangle):
        tabSin_[n1] = sin(ang * irho)
        tabCos_[n1] = cos(ang * irho)
        ang += theta

    # ###############################################################################################
    # i_s, j_s = np.where(img > 0)
    cdef int i, j, n, r
    for i in range(height):
        for j in range(width):
            if img_[i, j] != 0:
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
