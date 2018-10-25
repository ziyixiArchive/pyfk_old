from functools import partial
from multiprocessing import Pool

import numba
import numpy as np
from scipy.special import jv

import pyfk.gf.kernel as fkkernel
import pyfk.gf.setting as setting


def waveformIntegration(sum):
    aj0list, aj1list, aj2list = calbessel()
    sum = maincal(sum, aj0list, aj1list, aj2list)
    return sum


def calbessel():
    nfft2 = setting.nfft2
    dw = setting.dw
    pmin = setting.pmin
    dk = setting.dk
    kc = setting.kc
    pmax = setting.pmax
    x = setting.x
    wc1 = setting.wc1

    z = pmax*nfft2*dw/kc
    k = np.sqrt(z**2+1)
    kc = kc**2

    besseltemp = [[] for j in range(wc1-1, nfft2)]
    comparetemp = np.array([0 for j in range(wc1-1, nfft2)])

    for j in range(wc1-1, nfft2):
        omega = j*dw
        j = j-(wc1-1)
        k = omega*pmin+0.5*dk
        n = (np.sqrt(kc+(pmax*omega)**2)-k)/dk
        comparetemp[j] = int(n)
        temp = np.zeros((3, int(n), np.shape(x)[0]))
        klist = np.zeros((np.shape(x)[0], int(n)))
        klist[:, :] = np.array([k+i*dk for i in range(int(n))])
        xlist = np.diag(x)
        zzlist = klist.T@xlist
        temp[0, :, :] = jv(0, zzlist)
        temp[1, :, :] = jv(1, zzlist)
        temp[2, :, :] = jv(2, zzlist)

        besseltemp[j] = temp
    row = np.shape(comparetemp)[0]
    column = int(np.max(comparetemp))
    aj0list, aj1list, aj2list = [
        np.zeros((nfft2-wc1+1, column, np.shape(x)[0])) for i in range(3)]
    for j in range(row):
        aj0list[j, :comparetemp[j], :] = besseltemp[j][0, :, :]
        aj1list[j, :comparetemp[j], :] = besseltemp[j][1, :, :]
        aj2list[j, :comparetemp[j], :] = besseltemp[j][2, :, :]
    return aj0list, aj1list, aj2list


def maincal(sum, aj0list, aj1list, aj2list):
    nfft2 = setting.nfft2
    wc1 = setting.wc1
    result = np.empty(nfft2-wc1+1, dtype=np.object)
    # start=time.time()
    # last=start
    # for j in range(wc1-1, nfft2):
    #     result[j-wc1+1]=maincal_raw(j,sum,aj0list,aj1list,aj2list)
    #     print(time.time()-start,time.time()-last)
    #     last=time.time()
    result[0] = maincal_raw(wc1-1, sum, aj0list, aj1list, aj2list)
    with Pool() as pool:
        result[1:] = pool.map(partial(maincal_raw, sum=sum, aj0list=aj0list,
                                      aj1list=aj1list, aj2list=aj2list), range(wc1, nfft2))
    for j, value in zip(range(wc1-1, nfft2), result):
        sum[:, :, j] = value[:, :, j]
    return sum


# below is designed for numba
@numba.njit
def maincal_raw_numba(j, sum, aj0list, aj1list, aj2list,
                      nfft2, dw, pmin, dk, kc, pmax, wc1, a, b, qa, qb, mu, d, si,
                      sigma, x, flip, theconst, dynamic, wc2, t0, nCom, taper, mb, wc,
                      ka, kb, u, aaa, bbb, ccc, eee, ggg, zzz, sss, temppp):
    ztemp = pmax*nfft2*dw/kc
    k = np.sqrt(ztemp**2+1)
    kc = kc**2
    omega = j*dw
    w = np.complex(omega, -sigma)

    ka[:] = 0
    kb[:] = 0
    att = np.log(w/(2*np.pi))/np.pi+np.complex(0., 0.5)
    ka = w/(a*(1.+att/qa))
    kb = w/(b*(1.+att/qb))
    ka = ka**2
    kb = kb**2

    k = omega*pmin+0.5*dk
    n = (np.sqrt(kc+(pmax*omega)**2)-k)/dk
    for i in range(int(n)):
        u[:, :] = 0
        # initial values
        aaa[:, :] = 0
        bbb[:, :] = 0
        ccc[:, :] = 0
        eee[:] = 0
        ggg[:] = 0
        zzz[:, :] = 0
        sss[:, :] = 0
        temppp[:, :] = 0
        # end initial values
        u = fkkernel.kernel(k, u, ka, kb,   aaa, bbb, ccc,
                            eee, ggg, zzz, sss, temppp, mu, d, si+0j)
        aj0 = aj0list[j - wc1+1, i, :]
        aj1 = aj1list[j - wc1+1, i, :]
        aj2 = aj2list[j - wc1+1, i, :]
        z = k*x

        sum[:, 0, j] += u[0, 0]*aj0*flip
        sum[:, 1, j] += -u[0, 1]*aj1
        sum[:, 2, j] += -u[0, 2]*aj1

        nf = (u[1, 1]+u[1, 2])*aj1/z
        sum[:, 3, j] += u[1, 0]*aj1*flip
        sum[:, 4, j] += u[1, 1]*aj0-nf
        sum[:, 5, j] += u[1, 2]*aj0-nf

        nf = 2.*(u[2, 1]+u[2, 2])*aj2/z
        sum[:, 6, j] += u[2, 0]*aj2*flip
        sum[:, 7, j] += u[2, 1]*aj1-nf
        sum[:, 8, j] += u[2, 2]*aj1-nf

        k = k+dk
    filter = theconst
    if(dynamic and j+1 > wc):
        filter = 0.5*(1.+np.cos((j+1-wc)*taper))*filter
    if(dynamic and j+1 < wc2):
        filter = 0.5*(1.+np.cos((wc2-j-1)*np.pi/(wc2-wc1)))*filter

    phi = omega*t0
    atttemp = filter*(np.cos(phi) + np.sin(phi)*(1j))
    for bbq in range(nCom):
        sum[:, bbq, j] *= atttemp
    return sum


def maincal_raw(j, sum, aj0list, aj1list, aj2list):
    nfft2 = setting.nfft2
    dw = setting.dw
    pmin = setting.pmin
    dk = setting.dk
    kc = setting.kc
    pmax = setting.pmax
    wc1 = setting.wc1
    a = setting.a
    b = setting.b
    qa = setting.qa
    qb = setting.qb
    sigma = setting.sigma
    x = setting.x
    flip = setting.flip
    theconst = setting.theconst
    dynamic = setting.dynamic
    wc2 = setting.wc2
    t0 = setting.t0
    nCom = setting.nCom
    taper = setting.taper
    mb = setting.mb
    wc = setting.wc
    mu = setting.mu
    d = setting.d
    si = setting.si

    ka = np.zeros(mb, dtype=np.complex)
    kb = np.zeros(mb, dtype=np.complex)
    u = np.zeros((3, 3), dtype=np.complex)
    aaa = np.zeros((5, 5), dtype=np.complex)
    bbb = np.zeros((7, 7), dtype=np.complex)
    ccc = np.zeros((7, 7), dtype=np.complex)
    eee = np.zeros(7, dtype=np.complex)
    ggg = np.zeros(7, dtype=np.complex)
    zzz = np.zeros((3, 5), dtype=np.complex)
    sss = np.zeros((3, 6), dtype=np.complex)
    temppp = np.zeros((4, 4), dtype=np.complex)

    sum = maincal_raw_numba(j, sum, aj0list, aj1list, aj2list,
                            nfft2, dw, pmin, dk, kc, pmax, wc1, a, b, qa, qb, mu, d, si,
                            sigma, x, flip, theconst, dynamic, wc2, t0, nCom, taper, mb, wc,
                            ka, kb, u, aaa, bbb, ccc, eee, ggg, zzz, sss, temppp)
    return sum
