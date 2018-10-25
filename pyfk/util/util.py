import numba
import numpy as np

@numba.njit
def real(x):
    return (x+np.conj(x))/2.

@numba.njit
def imag(x):
    return (x-np.conj(x))/2.