import numpy as np
from obspy.core import UTCDateTime
from obspy.core.trace import Trace

import pyfk.gf.setting as setting
import pyfk.gf.source as fksource
import pyfk.gf.waveformintegration as waveformIntegration
import pyfk.util.ffr as fkffr


class MainFK_input(object):
    def __init__(self, input):
        self.nlay = input['mb']
        self.ndis = np.shape(input['x'])[0]
        self.nt = int(input['nfft']*input['smth']/2)

        setting.initModel(self.nlay, self.ndis, self.nt)
        self.synafter(setting.model_namelist)
        # mb,src,stype,rcv,updn
        self.mb = input['mb']
        self.src = input['src']
        self.stype = input['stype']
        self.rcv = input['rcv']
        self.updn = input['updn']

        # d(j),a(j),b(j),rho(j),qa(j),qb(j)
        self.d = input['d']
        self.a = input['a']
        self.b = input['b']
        self.rho = input['rho']
        self.qa = input['qa']
        self.qb = input['qb']

        # sigma,nfft,dt,taper,tb,smth,wc1,wc2
        self.sigma = input['sigma']
        self.nfft = input['nfft']
        self.dt = input['dt']
        self.taper = input['taper']
        self.tb = input['tb']
        self.smth = input['smth']
        self.wc1 = input['wc1']
        self.wc2 = input['wc2']

        # pmin,pmax,dk,kc
        self.pmin = input['pmin']
        self.pmax = input['pmax']
        self.dk = input['dk']
        self.kc = input['kc']

        # x(ix),t0(ix)
        self.x = np.array(input['x'])
        self.t0 = np.array(input['t0'])
        self.sac_com = input['sac_com']

        # addition
        self.check_input()

    def check_input(self):
        if(self.mb > self.nlay or self.src == self.rcv):
            raise Exception(str(self.mb)+'>'+str(self.nlay) +
                            ' or source receiver at same level')

    def synbefore(self, namelist):
        for item in namelist:
            exec('setting.'+item+'=self.'+item)

    def synafter(self, namelist):
        for item in namelist:
            exec('self.'+item+'=setting.'+item)


class MainFK(MainFK_input):
    def __init__(self, input):
        super(MainFK, self).__init__(input)

    def inputAna(self):
        nCom = 3 + 3*self.stype
        flip = 1
        if(self.rcv > self.src):
            flip = -1
            self.src = self.mb - self.src + 2
            self.rcv = self.mb - self.rcv + 2

        hs = 0.
        if(flip < 0):
            self.d = self.d[::-1]
            self.a = self.a[::-1]
            self.b = self.b[::-1]
            self.rho = self.rho[::-1]
            self.qa = self.qa[::-1]
            self.qb = self.qb[::-1]

        self.b[self.b < self.epsilon] = self.epsilon

        self.mu = self.rho*self.b*self.b
        self.xi = self.b*self.b/(self.a*self.a)

        for i, value in enumerate(self.d):
            if(i < self.src-1 and i >= self.rcv-1):
                hs += value
        vs = self.b[self.src-1]
        self.si = fksource.source(self.stype, self.xi[self.src-1],
                                  self.mu[self.src-1], self.si, flip)
        dynamic = True
        nfft2 = 0
        wc1 = 0
        if(self.nfft == 1):
            dynamic = False
            nfft2 = 1
            wc1 = 1
        else:
            nfft2 = self.nfft/2
        nfft2 = int(nfft2)

        dw = np.pi*2/(self.nfft*self.dt)
        self.sigma = self.sigma*dw/(np.pi*2)
        wc = int(nfft2*(1.-self.taper))
        if(wc < 1):
            wc = 1
        self.taper = np.pi/(nfft2-wc+1)
        if(self.wc2 > wc):
            self.wc2 = wc
        if(self.wc1 > self.wc2):
            self.wc1 = self.wc2

        self.kc = self.kc/hs
        self.pmin = self.pmin/vs
        self.pmax = self.pmax/vs

        xmax = np.max([hs, np.max(self.x)])
        self.t0 = self.t0-self.tb*self.dt
        self.dk = self.dk*np.pi/xmax
        theconst = self.dk/(np.pi*2)

        self.nfft2 = nfft2
        self.dw = dw
        self.flip = flip
        self.theconst = theconst
        self.dynamic = dynamic
        self.wc = wc
        self.nCom = nCom

    def waveformIntegration_wrap(self, sum):
        setting.init_mainfk_waveformIntegration()

        self.synbefore(setting.model_namelist)
        self.synbefore(setting.namelist_forwaveformIntegration)
        sum = waveformIntegration.waveformIntegration(sum)

        return sum

    def myffr(self, sum):
        self.dt /= self.smth
        self.nfft *= self.smth
        nfft3 = int(self.nfft/2)
        dfac = np.exp(self.sigma*self.dt)
        data = np.zeros(self.nt, dtype=np.complex)
        tdata = np.zeros(2*self.nt)
        result = np.zeros((self.ndis, self.nCom, 2*self.nt))
        resulthead = np.zeros((self.ndis, self.nCom), dtype=np.object)

        for ix in range(self.ndis):
            for l in range(self.nCom):
                data[:self.nfft2] = sum[ix, l, :]
                data[self.nfft2:nfft3] = 0
                data = fkffr.fftr(data, nfft3, -self.dt)
                z = np.exp(self.sigma*self.t0[ix])
                for j in range(1, nfft3+1):
                    tdata[2*j-2] = data[j-1].real*z
                    z *= dfac
                    tdata[2*j-1] = data[j-1].imag*z
                    z *= dfac
                head = {'delta': self.dt,
                        'starttime': UTCDateTime(self.t0[ix]),
                        'sac': {
                            'dist': self.x[ix], 't1': self.sac_com[ix]['tp'], 't2': self.sac_com[ix]['ts'],
                            'user1': self.sac_com[ix]['pa'], 'user2': self.sac_com[ix]['sa']}}
                result[ix, l, :] = tdata
                resulthead[ix, l] = head
        return result, resulthead

    def run(self):
        self.inputAna()

        sum = np.zeros((setting.ndis, 9, setting.nt), dtype=np.complex)
        sum = self.waveformIntegration_wrap(sum)
        result, resulthead = self.myffr(sum)
        wave = np.zeros((self.ndis, self.nCom), dtype=np.object)
        for ix in range(self.ndis):
            for l in range(self.nCom):
                wave[ix, l] = Trace(header=resulthead[ix, l],
                                    data=result[ix, l, :])
        return wave
