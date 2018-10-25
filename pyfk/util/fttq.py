import numpy as np
import numba
import pyfk.util.ffr as fkffr

@numba.njit
def fttq(dt,ts,ncount,f):
    nfft=2048
    nfft2=nfft/2
    dw=np.pi/(nfft2*dt)
    w=dw
    f[0]=1
    f[1]=0
    for i in range(1,nfft2):
        a=w*(ts*np.log(w)/np.pi-nfft2*dt)
        ep=np.exp(-0.5*w*ts)
        f[2*i]=ep*np.cos(a)
        f[2*i+1]=ep*np.sin(a)
        w+=dw
    fkffr.fftr(f,nfft2,-dt)
    # C cut f(t) so that only values larger enough are returned
    a=0
    icount=0
    for i in range(nfft):
        icount=i
        f[i]*=dt
        a+=f[i]
        if(a>0.02):
            break
    f[0]=f[icount]
    for n in range(1,nfft-icount):
        ncount=n
        f[n]=f[n+icount-1]*dt
        a+=f[n]
        if(a>0.98):
            break
    ncount+=1