description=r'''Compute displacements in cm in the up, radial, and transverse (clockwise) directions produced by difference seismic sources
author: Ziyi Xi， 04/05/2018， USTC.
Adapt from: Lupei Zhu, 02/15/2005, SLU.

Examples:
* To compute three-component velocity at N33.5E azimuth from a Mw 4.5 earthquake (strike 355, dip 80, rake -70), use:
    syn -M 4.5/355/80/-70 -D 1 -A 33.5 -O PAS.z -G hk_15/50.grn.0
* To compute displacement from an explosion, use:
   	syn -M 3.3e20 -D 1 -A 33.5 -O PAS.z -G hk_15/50.grn.a
  or:
    syn -M3.3e20/1/0/0/1/0/1 -D1 -A33.5 -OPAS.z -Ghk_15/50.grn.0
'''

helpM=r'''Specify source magnitude and orientation or moment-tensor
For double-couple, mag is Mw, strike/dip/rake are in A&R convention
For explosion; mag in in dyne-cm, no strike, dip, and rake needed
For single-force source; mag is in dyne, only strike and dip are needed
For moment-tensor; mag in dyne-cm, x=N,y=E,z=Down
'''

helpA=r'''Set station azimuth in degree measured from the North'''

helpS=r'''Specify the SAC file name of the source time function (its sum. must be 1)'''

helpD=r'''Specify the source time function as a trapezoid,
give the total duration and rise-time (0-0.5, default 0.5=triangle)'''

helpF=r'''apply n-th order Butterworth band-pass filter, SAC lib required (off, n=4, must be < 10)'''

helpI=r'''Integration once'''

helpJ=r'''Differentiate the synthetics'''

helpO=r'''Output SAC file name'''

helpQ=r'''Convolve a Futterman Q operator of tstar (no)'''

