import numba 
import numpy as np 
import pyfk.gf.setting as setting

@numba.njit
def kernel(k,u,ka,kb,a,bb,c,e,g,z,ss,temp,mu,d,si):
    # initial values in model namespace
    mb=setting.mb
    src=setting.src
    stype=setting.stype
    updn=setting.updn
    epsilon=setting.epsilon
    ndis=setting.ndis
    nt=setting.nt
    rcv=setting.rcv
    # end initial values in model namespace

    # begin initial values in layer namespace
    kd,exa,exb,mu2,y,y1=[0. for i in range(6)]
    ra,rb,Ca,Cb,Ya,Xa,Yb,Xb,r,r1=[0+0j for i in range(10)]
    # end initial values in layer namespace



    # c Explanations:
    # c a --- 4x4 p-sv Haskell matrix and a(5,5)=exb.
    # c b --- product of compound matrices from the receiver to the surface.
    # c c --- compound matrix.
    # c e --- vector, the first 5 members are E0|_{12}^{ij}, ij=12,13,23,24,34;
    # c       the last two are the 1st column of the 2x2 SH E0 (or unit vector if the top is elastic).
    # c g --- vector containing the Rayleigh and Love denominators. It is initialized in the
    # c       bottom half-space with  (E^-1)|_{ij}^{12}, ij=12,13,23,24,34, and
    # c       the 1st row of the 2x2 SH E^-1 (or a unit vector if the bottom is vacume).
    # c z --- z(n,j)=s(i)*X|_{ij}^{12} for p-sv and s(i)*X_5i for sh.

    #begin initialB
    # c***************************************************************
    # c Initialize b as an unit matrix; e as an unit vector;
    # c e = (1 0 0 0 0 1 0) for top halfspace boundary condition;
    # c g = (0 0 0 0 1 0 1) for free bottom boundary condition.
    # c***************************************************************
    for i in range(7):
        bb[i,i]=1
    e[0],e[5],g[4],g[6]=1,1,1,1
    # c propagation - start from the bottom   
    for j in np.arange(mb)[::-1]: 

        #begin layerParameter(k, lay)
        lay=j
        k2=k**2
        kka = ka[lay]/k2
        kkb = kb[lay]/k2
        r = 2/kkb
        kd = k*d[lay]
        mu2 = 2.*mu[lay]
        ra = np.sqrt(1. - kka)
        rb = np.sqrt(1. - kkb)
        r1 = 1. - 1./r
        #end layerParameter(k, lay)

        if(j==mb-1 and d[j]<epsilon):
            # begin initialG
            # c***************************************************************
            # c Initialize the g row-vector. The first 5 elements are the
            # c inverse(E)|_{ij}^{12}, ij=12,13,23,24,34.
            # c g14 is omitted because g14 = -g23.
            # c The last two are the 5th row of E^-1.
            # c***************************************************************

            # c p-sv, see EQ 33 on ZR/p623, constant omitted.
            delta  = r*(1.-ra*rb)-1.
            g[0] = mu2*(delta-r1)
            g[1] = ra
            g[2] = delta
            g[3] =-rb
            g[4] = (1.+delta)/mu2

            # c sh, use the 5th row of E^-1, see EQ A4 on ZR/p625, 1/2 omitted
            g[5]=-1. 
            g[6]=2./(rb*mu2)
            # end initailG

        elif(j==0 and d[0]<epsilon):
            # begin eVector(e)
            # c***************************************************************
            # c The first 5 members are E|_12^ij, ij=12,13,23,24,34.
            # c The last two are the first column of SH E matrix.
            # c***************************************************************

            # c For p-sv, compute E|_(12)^(ij), ij=12, 13, 23, 24, 34.
            e[0] = ra*rb-1.
            e[1] = mu2*rb*(1.-r1)
            e[2] = mu2*(r1-ra*rb)
            e[3] = mu2*ra*(r1-1.)
            e[4] = mu2*mu2*(ra*rb-r1*r1)
            # c sh part
            e[5]=-1. 
            e[6]=mu2*rb/2. 
            # end eVector(e)
            break

        else:
            # begin compoundMatrix(c)
            # c***************************************************************
            # c The upper-left 5x5 is the 6x6 compound matrix of the P-SV Haskell matrix,
            # c       a(ij,kl) = A|_kl^ij, ij=12,13,14,23,24,34,
            # c after dropping the 3rd row and colummn and replacing the 4th row
            # c by (2A41, 2A42, 2A44-1,2A45,2A46) (see W&H, P1035).
            # c The lower-right c 2x2 is the SH part of the Haskell matrix.
            # c Input: layer parameters passed in by /layer/.
            # c Output: compound matrix a, scaled by exa*exb for the P-SV and exb for the SH.
            # c***************************************************************
            Ca,Ya,Xa,exa=sh_ch(Ca,Ya,Xa,exa,ra,kd)
            Cb,Yb,Xb,exb=sh_ch(Cb,Yb,Xb,exb,rb,kd)

            CaCb=Ca*Cb
            CaYb=Ca*Yb
            CaXb=Ca*Xb
            XaCb=Xa*Cb
            XaXb=Xa*Xb
            YaCb=Ya*Cb
            YaYb=Ya*Yb
            ex = exa*exb
            r2 = r*r
            r3 = r1*r1

            # c p-sv, scaled by exa*exb to supress overflow
            c[0,0] = ((1.+r3)*CaCb-XaXb-r3*YaYb-2.*r1*ex)*r2
            c[0,1] = (XaCb-CaYb)*r/mu2
            c[0,2] = ((1.+r1)*(CaCb-ex)-XaXb-r1*YaYb)*r2/mu2
            c[0,3] = (YaCb-CaXb)*r/mu2
            c[0,4] = (2.*(CaCb-ex)-XaXb-YaYb)*r2/(mu2*mu2)

            c[1,0] = (r3*YaCb-CaXb)*r*mu2
            c[1,1] = CaCb
            c[1,2] = (r1*YaCb-CaXb)*r
            c[1,3] =-Ya*Xb
            c[1,4] = c[0,3]

            c[2,0] = 2.*mu2*r2*(r1*r3*YaYb-(CaCb-ex)*(r3+r1)+XaXb)
            c[2,1] = 2.*r*(r1*CaYb-XaCb)
            c[2,2] = 2.*(CaCb - c[0,0]) + ex
            c[2,3] =-2.*c[1,2]
            c[2,4] =-2.*c[0,2]

            c[3,0] = mu2*r*(XaCb-r3*CaYb)
            c[3,1] =-Xa*Yb
            c[3,2] =-c[2,1]/2.
            c[3,3] = c[1,1]
            c[3,4] = c[0,1]

            c[4,0] = mu2*mu2*r2*(2.*(CaCb-ex)*r3-XaXb-r3*r3*YaYb)
            c[4,1] = c[3,0]
            c[4,2] =-c[2,0]/2.
            c[4,3] = c[1,0]
            c[4,4] = c[0,0]

            # c sh, scaled by exb
            c[5,5] = Cb
            c[5,6] =-2.*Yb/mu2
            c[6,5] =-mu2*Xb/2.
            c[6,6] = Cb
            # end compoundMatrix(c)

            #begin propagateG(c, g)
            # c***************************************************************
            # c propagate g vector upward using the compound matrix
            # c       g = g*a
            # c***************************************************************
            g=g@c
            #end propagateG(c, g)

        if(j==src-1):
            # begin separatS
            if(updn==0):
                ss[:stype+1,:]=si[:stype+1,:]
            else:
                ra1 = 1./ra
                rb1 = 1./rb
                dum = updn*r

                temp[0,0] = 1.
                temp[0,1] = dum*(rb-r1*ra1)
                temp[0,2] = 0
                temp[0,3] = dum*(ra1-rb)/mu2
                temp[1,0] = dum*(ra-r1*rb1)
                temp[1,1] = 1.
                temp[1,2] = dum*(rb1-ra)/mu2
                temp[1,3] = 0
                temp[2,0] = 0
                temp[2,1] = dum*(rb-r1*r1*ra1)*mu2
                temp[2,2] = 1.
                temp[2,3] = dum*(r1*ra1-rb)
                temp[3,0] = dum*(ra-r1*r1*rb1)*mu2
                temp[3,1] = 0
                temp[3,2] = dum*(r1*rb1-ra)
                temp[3,3] = 1.
                temp_sh   = (updn*2/mu2)*rb1

                ss[:stype+1,:4]=(si[:stype+1,:4]@temp.T)/2
                ss[:stype+1,4]=(si[:stype+1,4] + temp_sh*si[:stype+1,5])/2
                ss[:stype+1,5]=(si[:stype+1,5]+si[:stype+1,4]/temp_sh)/2
            # end separatS

            # begin initialZ(ss, g, z)
            # c***************************************************************
            # c initialize the row-vector z at the source z(j)=s(i)*X|_ij^12
            # c for P-SV and z(j)=s(i)*X(5,i) for SH.
            # c  input:
            # c       s(3,6)  ---- source coef. for n=0,1,2
            # c       g(7)    ---- g vector used to construct matrix X|_ij^12
            # c                    |  0   g1  g2 -g3 |
            # c        X|_ij^12 =  | -g1  0   g3  g4 | for P-SV.
            # c                    | -g2 -g3  0   g5 |
            # c                    |  g3 -g4 -g5  0  |
            # c        X(5,i) = ( g6 g7 )     for SH.
            # c  output:
            # c       z(3,5)  ---- z vector for n=0,1,2
            # c***************************************************************
            
            # c for p-sv, see WH p1018
            z[:,0]=-ss[:,1]*g[0]-ss[:,2]*g[1]+ss[:,3]*g[2]
            z[:,1]= ss[:,0]*g[0]-ss[:,2]*g[2]-ss[:,3]*g[3]
            z[:,2]= ss[:,0]*g[1]+ss[:,1]*g[2]-ss[:,3]*g[4]
            z[:,3]=-ss[:,0]*g[2]+ss[:,1]*g[3]+ss[:,2]*g[4]
            # c for sh
            z[:,4]= ss[:,4]*g[5]+ss[:,5]*g[6]
            # end initialZ(ss, g, z)

        if(j<src-1):
            if(j>=rcv-1):
                # begin haskellMatrix(a)
                # c***************************************************************
                # c compute 4x4 P-SV Haskell a for the layer, the layer parameter
                # c is passed in by common /layer/.
                # c***************************************************************
                Ca = Ca*exb
                Xa = Xa*exb
                Ya = Ya*exb
                Cb = Cb*exa
                Yb = Yb*exa
                Xb = Xb*exa
                # c p-sv, scaled by exa*exb, see p381/Haskell1964 or EQ 17 of ZR
                a[0,0] = r*(Ca-r1*Cb)
                a[0,1] = r*(r1*Ya-Xb)
                a[0,2] = (Cb-Ca)*r/mu2
                a[0,3] = (Xb-Ya)*r/mu2

                a[1,0] = r*(r1*Yb-Xa)
                a[1,1] = r*(Cb-r1*Ca)
                a[1,2] = (Xa-Yb)*r/mu2
                a[1,3] =-a[0,2]

                a[2,0] = mu2*r*r1*(Ca-Cb)
                a[2,1] = mu2*r*(r1*r1*Ya-Xb)
                a[2,2] = a[1,1]
                a[2,3] =-a[0,1]

                a[3,0] = mu2*r*(r1*r1*Yb-Xa)
                a[3,1] =-a[2,0]
                a[3,2] =-a[1,0]
                a[3,3] = a[0,0]

                # c sh, the Haskell matrix is not needed. it is replaced by exb
                a[4,4] = exb
                # end haskellMatrix(a)


                #begin propagateZ(a, z)
                z[:,:]=z@a
                # z[:,:]=1+0j
                #end propagateZ(a, z)
            else:
                # begin propagateB(c, b)
                bb[:,:]=bb@c
                # end propagateB(c, b)

    # c add the top halfspace boundary condition   
    e[2]=2*e[2]
    rayl = np.sum(g[:5]*e[:5])
    love = np.sum(g[5:]*e[5:])

    g[:4]=(bb[:4,:5]@e[:5])
    g[2]/=2
    g[5]=bb[5,5]*e[5]+bb[5,6]*e[6]
    for i in range(3):
        dum    = z[i,1]*g[0]+z[i,2]*g[1]-z[i,3]*g[2]
        z[i,1] =-z[i,0]*g[0]+z[i,2]*g[2]+z[i,3]*g[3]
        z[i,0] = dum
        z[i,4] = z[i,4]*g[5]

    # c displacement kernels at the receiver
    dum=k   
    if(stype==1):
        dum=1
    u[:,0]=dum*z[:,1]/rayl
    u[:,1]=dum*z[:,0]/rayl
    u[:,2]=dum*z[:,4]/love

    return u



@numba.njit
def sh_ch(c,y,x,ex,a,kd):
    # c***************************************************************
    # c compute c=cosh(a*kd); y=sinh(a*kd)/a; x=sinh(a*kd)*a
    # c and multiply them by ex=exp(-Real(a*kd)) to supress overflow
    # c
    # c called by: compoundMatrix()           in compound.f
    # c***************************************************************
    y = kd*a
    r = y.real
    i = y.imag
    ex =np.exp(-r)
    y = 0.5*np.complex(np.cos(i),np.sin(i))
    x = ex*ex*np.conj(y)
    c = y + x
    x = y - x
    y = x/a
    x = x*a
    return c,y,x,ex



