import numpy as np

class SYNvalues(object):
    def __init__(self):
        self.cmpaz=np.zeros(3)
        self.dura=0
        self.rise=0
        self.f1=0
        self.f2=0
        self.order=0
        self.m0=0
        self.mt=np.zeros((2,3))
        self.ns=0
        self.shift=0
        self.intg=0
        self.diff=0
        self.tstar=0
        self.nn=0
        self.rad=np.zeros((4,3))
        self.az=0
        self.outsrc=False
