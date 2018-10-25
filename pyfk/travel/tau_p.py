import numba
import numpy as np
from numba import float64, void

import pyfk.travel.setting as setting


@numba.njit
def findp0(x, p0, topp, bttm, ray_len, vps, thk):
    zero = 1.e-7
    p1 = np.complex(0., p0.imag)
    while(p1 != p0):
        p2 = p0
        p0 = 0.5*(p1+p2)
        dtdp0 = dtdp(x, p0, topp, bttm, ray_len, vps, thk)
        if(np.abs(dtdp0) < zero or p0 == p1 or p0 == p2):
            return p0
        if(dtdp0.real > 0):
            p1 = p0
            p0 = p2
    return p0


@numba.njit
def taup(p, x, topp, bttm, ray_len, vps, thk):
    result = p*x
    pp = p*p
    result += np.sum(np.sqrt(vps[0, topp-1:bttm]-pp)*ray_len[0, topp-1:bttm] +
                     np.sqrt(vps[1, topp-1:bttm]-pp) *
                     ray_len[1, topp-1:bttm])
    return result


@numba.njit
def dtdp(x, p, topp, bttm,  ray_len, vps, thk):
    pp = p*p
    result = 0.
    for j in range(topp-1, bttm):
        temp = ray_len[0, j]/np.sqrt(vps[0, j]-pp) + \
            ray_len[1, j]/np.sqrt(vps[1, j]-pp)
        result = result-temp
    result = x+p*result
    return result


@numba.njit
def runtravel(xlist, t0, td, p0, pd, vps, thk):
    for i in range(len(list(xlist))):
        x = xlist[i]
        # direct arrival
        ray_len = np.zeros((2, setting.num_lay))
        topp = setting.rcv_lay+1
        bttm = setting.src_lay
        aa = 1.e20

        ray_len[0, topp-1:bttm] = thk[topp-1:bttm]
        ray_len[1, topp-1:bttm] = 0.
        aa = np.min(vps[0, topp-1:bttm])

        pd[i] = np.complex(np.sqrt(aa), 1.e-20)
        pd[i] = findp0(x, pd[i], topp, bttm, ray_len, vps, thk)
        temp = taup(pd[i], x, topp, bttm, ray_len, vps, thk)
        td[i] = temp.real
        t0[i] = td[i]
        p0[i] = pd[i]
        # reflected arrivals from below.
        therange = np.arange(setting.src_lay+1, setting.num_lay)
        for bttm in therange:
            ray_len[0, bttm-1] = 2.*thk[bttm-1]
            ray_len[1, bttm-1] = 0.
            # aa = np.min([aa, vps[0, bttm-1]])
            if(aa > vps[0, bttm-1]):
                aa = vps[0, bttm-1]
            p = np.complex(np.sqrt(aa), 1.e-20)
            p = findp0(x, p, topp, bttm, ray_len, vps, thk)
            # aa = np.min([aa, vps[0, bttm]])
            if(aa > vps[0, bttm]):
                aa = vps[0, bttm]
            pc = np.complex(np.sqrt(aa), 1.e-20)
            if(p.real > pc.real):
                p = pc
            temp = taup(p, x, topp, bttm, ray_len, vps, thk)
            t = temp.real
            if(t < t0[i]):
                t0[i] = t
                p0[i] = p
        # reflected arrivals from above.
        bttm = setting.src_lay

        therange = np.arange(setting.rcv_lay, 0, -1)
        for topp in therange:
            if(topp == 0):
                break
            ray_len[0, topp-1] = 2.*thk[topp-1]
            ray_len[1, topp-1] = 0.
            # aa = np.min([aa, vps[0, topp-1]])
            if(aa > vps[0, topp-1]):
                aa = vps[0, topp-1]
            p = np.complex(np.sqrt(aa), 1.e-20)
            p = findp0(x, p, topp, bttm, ray_len, vps, thk)
            if(topp > 1):
                # aa = np.min([aa, vps[0, topp-2]])
                if(aa > vps[0, topp-2]):
                    aa = vps[0, topp-2]
            pc = np.complex(np.sqrt(aa), 1.e-20)
            if(p.real > pc.real):
                p = pc
            temp = taup(p, x, topp, bttm, ray_len, vps, thk)
            t = temp.real
            if(t < t0[i]):
                t0[i] = t
                p0[i] = p
