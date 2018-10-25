from obspy.core.trace import Trace

import numpy as np
import obspy
from . import radiats
from pyfk.util.fttq import fttq


class MainSYN(object):
    def __init__(self,values):
        self.values=values

    def GreenFunction(self,greenfunctionlist=None):
        theshape=len(greenfunctionlist)
        # print(greenfunctionlist[0][0].stats)
        thenpts=greenfunctionlist[0][0].stats['npts']
        synlist=np.empty((np.shape(greenfunctionlist)[0],3),dtype=np.object)
        # print(Trace(header=greenfunctionlist[0,0].stats,data=np.zeros(thenpts)))
        for i in range(np.shape(greenfunctionlist)[0]):
            for j in range(3):
                synlist[i,j]=Trace(header=greenfunctionlist[i,0].stats,data=np.zeros(thenpts))
        # synlist=np.array([[Trace(header=greenfunctionlist[i][0].stats,data=np.zeros(thenpts)) for j in range(3)] for i in range(len(list()))])
        for ix in range(theshape):
            for i in range(self.values.nn):
                for j in range(3):
                    coef=self.values.m0
                    if(self.values.nn>1):
                        coef*=self.values.rad[i][j]
                    synlist[ix][j].data+=greenfunctionlist[ix][3*i+j].data*coef
        self.synlist=synlist 


        

    def SrcFunction(self,srcfunction=None):
        values=self.values
        if(values.outsrc):
            delta1=self.synlist[0][0].stats['delta']
            delta2=srcfunction.stats['delta']
            if(delta1!=delta2):
                raise Exception('delta in Green Function and triggering source should be equal!')
            self.src=srcfunction
        else:
            # print(self.synlist[0][0])
            dt=self.synlist[0][0].stats['delta']
            ns=int(values.dura/dt)
            if(ns<2):
                ns=2
            self.src=np.zeros(ns+1)
            nr=int(values.rise*ns)
            if(nr<1):
                nr=1
            if(2*nr>ns):
                nr=ns/2
            amp=1./(nr*(ns-nr))
            self.src[:nr]=amp*np.arange(nr)
            self.src[nr:ns-nr]=nr*amp
            self.src[ns-nr:]=(ns-np.arange(ns-nr,ns+1))*amp

    def run(self,greenfunctionlist):
        self.GreenFunction(greenfunctionlist)
        self.SrcFunction()

        theshape=np.shape(self.synlist[0][0].data)[0]
        thedelta=self.synlist[0][0].stats['delta']
        for i in range(len(self.synlist)):
            for j in range(3):
                self.synlist[i][j].data=np.convolve(self.src,self.synlist[i][j].data)[:theshape]
                if(self.values.intg):
                    self.synlist[i][j].data=np.cumsum(self.synlist[i][j])*thedelta
                if(self.values.diff):
                    self.synlist[i][j].data=np.hstack(([0],np.diff(self.synlist[i][j].data)/thedelta))
                if(not (self.values.f1==0 and self.values.f2==0)):
                    self.synlist[i][j]=self.synlist[i][j].filter("bandpass",freqmin=self.values.f1,freqmax=self.values.f2,corners=self.values.order)
                if(not (self.values.tstar is None)):    
                    if(self.values.tstar>0):
                        ftm=np.zeros(2048)
                        ncount=0
                        fttq.fttq(thedelta,self.values.tstar,ncount,ftm)
                        ftm=ftm[:ncount]
                        self.synlist[i][j].data=np.convolve(ftm,self.synlist[i][j].data)[:theshape]

    def output(self):
        cmpinc=[0.,90.,90.]
        cmpaz=self.values.cmpaz
        for i in range(len(self.synlist)):
            for j in range(3):
                # self.synlist[i][j].stats['sac']['b']-=self.values.shift
                # self.synlist[i][j].stats['sac']['e']-=self.values.shift
                self.synlist[i][j].stats['sac']['az']=self.values.az
                self.synlist[i][j].stats['sac']['cmpinc']=cmpinc[j]
                self.synlist[i][j].stats['sac']['cmpaz']=cmpaz[j]

        return self.synlist
