
helpN = r'''nt is the number of points, must be 2^n ({nft}).
Note that nt=1 will compute static displacements (require st_fk compiled).
          nt=2 will compute static displacements using the dynamic solution.
dt is the sampling interval ({dt} sec).
smth makes the final sampling interval to be dt/smth, must be 2^n ({smth}).
dk is the non-dimensional sampling interval of wavenumber ({dk}).
taper applies a low-pass cosine filter at fc=(1-taper)*f_Niquest ({taper}).'''

helpM = r'''model name and source depth in km. f triggers earth flattening (off), k indicates that the 3rd column is vp/vs ratio (vp).
model has the following format (in units of km, km/s, g/cm3):
thickness vs vp_or_vp/vs [rho Qs Qp]
rho=0.77 + 0.32*vp if not provided or the 4th column is larger than 20 (treated as Qs).
Qs=500, Qp=2*Qs, if they are not specified.
If the first layer thickness is zero, it represents the top elastic half-space.
Otherwise, the top half-space is assumed to be vacuum and does not need to be specified.
The last layer (i.e. the bottom half space) thickness should be always be zero.'''

helpD=r'''use degrees instead of km.'''

helpH=r'''apply a high-pass filter with a cosine transition zone between freq. f1 and f2 in Hz ({f1}/{f2})'''

helpP=r'''specify the min. and max. slownesses in term of 1/vs_at_the_source ({pmin}/{pmax}) and optionally kmax at zero frequency in term of 1/hs ({kmax}).'''

helpR=r'''receiver depth ({r_depth}).'''

helpS=r'''0=explosion; 1=single force; 2=double couple ({src}).'''

helpU=r'''1=down-going wave only; -1=up-going wave only ({updn}).'''

helpX=r'''dump the input to cmd for debug ({fk}).'''

description = r'''compute the Green's functions of a layered medium from
explosion, single-force, and double-couple sources.
author: Ziyi Xi， 04/05/2018， USTC.
Adapt from: Lupei Zhu, 02/15/2005, SLU. 

Examples
* To compute Green's functions up to 5 Hz with a duration of 51.2 s and at a dt of 0.1 s every 5 kms for a 15 km deep source in the HK model, use
fk.py -M hk/15/k -N 512/0.1 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80
* To compute static Green's functions for the same source, use
fk.py -M hk/15/k -N 2 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 > st.out
or use
fk.py -M hk/15/k -N 1 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 > st.out
* To compute Green's functions every 10 degrees for a 10 km deep source in the PREM model.
fk.py -M prem/10/f -D 10 20 30 40 50 60'''