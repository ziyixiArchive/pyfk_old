import numpy as np 
import numba

# /****************************************************************
#    compute horizontal radiation pattens for
#         double-couple   specified by az-strike,dip,rake
#         single force    specified by az-strike and dip
#         moment tnesor   specified by the moment tensor and az

# *****************************************************************/

# /*******************************************************************
# horizontal radiation coefficients of a double-couple, Haskell'64
# with tangential corrected

# INPUT:  az of the station measured from the strike of the fault clockwise,
#         dip, and rake of the fault-plane solution
# OUTPUT: rad[i][j] -> summation coef. for i-th azimuthal order and j-th component
#                         (0-> vertical, 1-> radial, 2->transverse)

# Algorithm:
#  V/R = f3n3*Z0
#      +((f1n3+f3n1)*cos(theta)+(f2n3+f3n2)*sin(theta))*Z1
#      +((f2n2-f1n1)*cos(2theta)+(-f1n2-f2n1)*sin(2theta))*Z2
#   T =-((f1n3+f3n1)*sin(theta)-(f2n3+f3n2)*cos(theta))*T1
#      -((f2n2-f1n1)*sin(2theta)-(-f1n2-f2n1)*cos(2theta))*T2

# where theta=pi/2-az.
#   n = (sin(delta),0,cos(delta))
#   F = (-sin(lamda)*cos(delta), cos(lamda), sin(lamda)*sin(delta))
# where delta is dip from the horizontal, lambda is the rake from the
# strike CCW.

# ********************************************************************/
DEG2RAD=1.745329252e-2
sin=np.sin 
cos=np.cos

@numba.njit
def dc_radiat(stk,dip,rak,rad):
    stk*=DEG2RAD
    dip*=DEG2RAD
    rak*=DEG2RAD

    sstk=sin(stk)
    cstk=cos(stk)
    sdip=sin(dip)
    cdip=cos(dip)
    srak=sin(rak)
    crak=cos(rak)
    sstk2=2*sstk*cstk
    cstk2=cstk*cstk-sstk*sstk
    sdip2=2*sdip*cdip
    cdip2=cdip*cdip-sdip*sdip

    rad[0][0]=0.5*srak*sdip2
    rad[0][1]=rad[0][0]
    rad[0][2]=0.
    rad[1][0]=-sstk*srak*cdip2+cstk*crak*cdip
    rad[1][1]=rad[1][0]
    rad[1][2]= cstk*srak*cdip2+sstk*crak*cdip
    rad[2][0]=-sstk2*crak*sdip-0.5*cstk2*srak*sdip2
    rad[2][1]=rad[2][0]
    rad[2][2]=cstk2*crak*sdip-0.5*sstk2*srak*sdip2

# /******************************************************
# horizontal radiation coefficients of a single-force
#    In:
#         stk: az_of_obs w.r.t to strike of the force
#                  measured clockwise
#         dip: dip of the force, from horizontal down

#    algorithm:
#         vertical (UP) = f3*Z0 + (f1*cos(theta)+f2*sin(theta))*Z1
#         radial  (OUT) = f3*R0 + (f1*cos(theta)+f2*sin(theta))*R1
#         tangen   (CW) =       - (f1*sin(theta)-f2*cos(theta))*T1
#     where F = (0,cos(dip),-sin(dip))
# ******************************************************/

@numba.njit
def sf_radiat(stk,dip,rad):
    stk*=DEG2RAD
    dip*=DEG2RAD
    sstk=sin(stk)
    cstk=cos(stk)
    sdip=sin(dip)
    cdip=cos(dip)

    rad[0][0]=-sdip
    rad[0][1]=rad[0][0]
    rad[0][2]=0.
    rad[1][0]=cdip*cstk
    rad[1][1]=rad[1][0]
    rad[1][2]=cdip*sstk

# /*****************************************************************
#  horizontal radiation coefficients from a moment-tensor m
#    see Jost and Herrmann, 1989 (note an error in Eq A5.4-A5.6)
# *****************************************************************/ 

@numba.njit
def mt_radiat(az,m,rad):
    az *= DEG2RAD
    saz = sin(az)
    caz = cos(az)
    saz2 = 2*saz*caz
    caz2 = caz*caz-saz*saz
    rad[2][0] = rad[2][1] = -0.5*(m[0][0]-m[1][1])*caz2 - m[0][1]*saz2
    rad[1][0] = rad[1][1] = -m[0][2]*caz - m[1][2]*saz
    rad[0][0] = rad[0][1] =  (2*m[2][2]-m[0][0]-m[1][1])/6.
    rad[2][2] = -0.5*(m[0][0]-m[1][1])*saz2 + m[0][1]*caz2
    rad[1][2] = -m[0][2]*saz + m[1][2]*caz
    rad[0][2] = 0.
    # /* contribution from explosion: */
    rad[3][0] = rad[3][1] = (m[0][0]+m[1][1]+m[2][2])/3.
    rad[3][2] = 0

    