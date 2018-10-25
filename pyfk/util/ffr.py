import numba
import numpy as np



@numba.njit
def fftr(x, n, dt):
    n2 = int(n/2)
    delw = np.pi/n
    isg = 1j
    if(dt > 0):
        delw = -delw
        isg = -isg
        x=fft(x, n, dt)
    x[0] = np.complex(x[0].real+x[0].imag, x[0].real-x[0].imag)

    w = delw
    for i in range(1, n2):
        j = n-i
        t = np.conj(x[j])
        g = x[i]+t
        h = x[i]-t
        # print(i,x[i],t,h,np.complex(np.cos(w), np.sin(w))*h,0.5*(g+isg*np.complex(np.cos(w), np.sin(w))*h))
        h = np.complex(np.cos(w), np.sin(w))*h
        x[i] = 0.5*(g+isg*h)
        x[j] = 0.5*(np.conj(g)+isg*np.conj(h))
        w += delw
        # print(i,x[i],g,isg,h,t,w)
    x[n2] = np.conj(x[n2])
    if(dt < 0.):
        x[0] = 0.5*x[0]
        x=fft(x, n, dt)

    return x


@numba.njit
def fft(a, n, dt):
    pi = -np.pi
    if(dt < 0.):
        pi = np.pi

    m = int(n/2)
    j = 0
    for i in range(1, n-1):
        k = m
        while(k <= j):
            j -= k
            k = int(k/2)
        j += k
        if(i < j):
            a[i], a[j] = a[j], a[i]
    m = 1
    step = 2
    while(m < n):
        u = 1+0j
        w = np.complex(np.cos(pi/m), np.sin(pi/m))
        j = 0
        while(j < m):
            i = j
            while(i < n):
                k = i+m
                t = a[k]*u
                a[k] = a[i]-t
                a[i] = a[i]+t
                i += step
            u = u*w
            j += 1
        m = step
        step *= 2
    if(dt < 0.):
        dt = -1./(n*dt)
    for i in range(n):
        a[i] = dt*a[i]
    return a
