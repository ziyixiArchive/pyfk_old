import numba
import numpy as np


@numba.njit
def source(stype, xi, mu, s, flip):
    s *= 0
    if(stype == 2):
        s[0, 1] = 2.*xi/mu
        s[0, 3] = 4.*xi-3.
        s[1, 0] = flip/mu
        s[1, 4] = -s[1, 0]
        s[2, 3] = 1.
        s[2, 5] = -1.
    elif(stype == 0):
        s[0, 1] = xi/mu
        s[0, 3] = 2.*xi
    elif(stype == 1):
        s[0, 2] = -flip
        s[1, 3] = -1.
        s[1, 5] = 1.
    return s
