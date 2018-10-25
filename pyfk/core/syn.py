
from pyfk.core.util import SYNvalues
import numpy as np
import pyfk.syn.radiats as radiats
from pyfk.syn.mainsyn import MainSYN  

class SYN(object):
    def __init__(self,model=None,azimuth=None,srcFunction=None,trapezoidshape=None,filter=None,intg=False,diff=False,tstar=None):
        self.model=model
        self.azimuth=azimuth
        self.srcFunction=srcFunction
        self.trapezoidshape=trapezoidshape
        self.filter=filter
        self.intg=intg
        self.diff=diff
        self.tstar=tstar
        
        self._getinformation()

    def _getinformation(self):
        self.values=SYNvalues()
        values=self.values

        # -A
        if(not (self.azimuth is None)):
            values.az=self.azimuth
            values.cmpaz[0]=0.
            values.cmpaz[1]=values.az
            values.cmpaz[2]=values.az+90
            if(values.cmpaz[2]>360.):
                values.cmpaz[2]-=360
        else:
            raise Exception('must provide azimuth!')

        # -D
        if(not(self.trapezoidshape is None)):
            thelist=list(self.trapezoidshape)
            values.outsrc=False
            if(len(thelist)==1):
                values.dura=thelist[0]
                values.rise=0.5
            elif(len(thelist)==2):
                values.dura=thelist[0]
                values.rise=thelist[1]
            else:
                raise Exception("wrong trapezoidshape input!")    

        # -F
        if(not(self.filter is None)):
            thelist=list(self.filter)
            if(len(thelist)==3):
                values.f1=thelist[0]
                values.f2=thelist[1]
                values.order=thelist[2]
            elif(len(thelist)==2):
                values.f1=thelist[0]
                values.f2=thelist[1]
                values.order=4
            else:
                raise Exception('wrong filter input!')
        else:
            values.f1=0
            values.f2=0

        # -M
        if(not(self.model is None)):
            thelist=np.array(self.model)
            nthelist=len(list(thelist))
            thelist=np.array(thelist)
            if(nthelist==1):
                values.m0=thelist[0]
                values.mt[:,:]=None
                values.nn=1
                values.m0*=1e-20
            elif(nthelist==3):
                values.m0=thelist[0]
                values.mt[0,0]=thelist[1]
                values.mt[0,1]=thelist[2]
                values.mt[0,2]=None
                values.mt[1,:]=None
                values.nn=2
                radiats.sf_radiat(values.az-values.mt[0][0],values.mt[0][1],values.rad)
                values.m0*=1.0e-15
            elif(nthelist==4):
                values.m0=thelist[0]
                values.mt[0,:]=thelist[1:]
                values.mt[1,:]=None
                values.nn=3
                radiats.dc_radiat(values.az-values.mt[0][0],values.mt[0][1],values.mt[0][2],values.rad)
                values.m0=np.power(10.,1.5*values.m0+16.1-20)
            elif(nthelist==7):
                values.m0=thelist[0]
                values.mt[0,:]=thelist[1:4]
                values.mt[1,:]=thelist[4:]
                values.nn=4
                radiats.mt_radiat(values.az,values.mt,values.rad)
                values.m0*=1.0e-20
            else:
                raise Exception('wrong model input!')

        # -S
        if(not(self.srcFunction is None)):
            tr=self.srcFunction
            self.ns=tr.stats.npts
            self.shift=-tr.stats.sac.b
            values.outsrc=True

        # -I -J -Q
        values.intg=self.intg
        values.diff=self.diff
        values.tstar=self.tstar

        self.values=values

    def run(self,greenfunctions):
        synobject=MainSYN(self.values)
        synobject.run(greenfunctions)
        return synobject.output()