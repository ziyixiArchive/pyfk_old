
import numpy as np


def initModel(nlay0, ndis0, nt0):
    global model_namelist
    model_array = {'d', 'rho', 'mu', 'xi', 'si', 'epsilon'}
    model_integer = {'mb', 'stype', 'src', 'rcv', 'nlay', 'ndis', 'nt', 'updn'}
    model_namelist = model_array | model_integer

    global d, rho, mu, xi, si, epsilon,  nlay, ndis, nt
    d = np.zeros(nlay0)
    rho = np.zeros(nlay0)
    mu = np.zeros(nlay0)
    xi = np.zeros(nlay0)
    si = np.zeros((3, 6))
    epsilon = 0.0001

    for item in model_integer:
        exec(item+'=0', globals())
    nlay = nlay0
    ndis = ndis0
    nt = nt0


def init_mainfk_waveformIntegration():
    global namelist_forwaveformIntegration
    namelist_forwaveformIntegration = {'wc1', 'nfft2', 'dw', 'sigma', 'pmin', 'dk', 'kc', 'pmax', 'x', 'flip', 'theconst',
                                       'dynamic', 'wc', 'wc2', 't0', 'nCom', 'a', 'b', 'qa', 'qb', 'taper'}

    for item in namelist_forwaveformIntegration:
        exec(item+'=0', globals())
