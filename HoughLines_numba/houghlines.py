import numpy as np
import cv2 as cv
from numba import jit

@jit
def HoughLines(img, rho, theta):
    height, width = img.shape
    numangle = int(np.pi/theta)
    numrho = int(((width + height) * 2 + 1) / rho)
    irho = 1 / rho

    accum = np.zeros((numangle + 2, numrho + 2), dtype=np.int32)
    tabSin = np.zeros((numangle, ), dtype=np.float32)
    tabCos = np.zeros((numangle, ), dtype=np.float32)
    
    ang = 0
    for n1 in range(numangle):
        tabSin[n1] = np.sin(ang * irho)
        tabCos[n1] = np.cos(ang * irho)
        ang += theta

    # ###############################################################################################
    for i in range(height):
        for j in range(width):
            if img[i, j] != 0:
                for n in range(numangle):
                    r = int(j * tabCos[n] + i * tabSin[n])
                    r += int((numrho - 1) / 2)
                    accum[n+1, r+1] += 1
    # ###############################################################################################
    ns, rs = np.where(accum > 0)
    lens = accum[ns, rs].reshape(-1, 1)
    ns = (ns - 1) * rho
    rs = (rs - (numrho-1)*0.5)*rho
    result = np.concatenate((rs.reshape(-1, 1), ns.reshape(-1, 1), lens), axis=1)

    return result
