from pyfk.gf.mainfk import MainFK
from pyfk.travel.travel import Travel
import numpy as np


class FK(object):
    def __init__(self, model=None, sdep=0, modeltype=None, distance=None, degrees=True, filter=None, number_of_points=256, dt=1, dk=0.3, smth=1, theproperty=np.array([0., 1., 15.]), rdep=0., srcType='dc', updn=0):
        if(model is None):
            raise Exception('must give a model')
        if(distance == None):
            raise Exception('must give the distance')

        self.default_values = {
            'r0': 6371.,
            'dt': 1	,  # sampling interval.
            'smth': 1,		# densify the output samples by a factor of smth.
            'nft': 256,		# number of points.
            'src': 2	,  # source type, 2=dc 1=sf 0=ex.
            'dk': 0.3,		# sampl. interval in wavenumber, in Pi/x, 0.1-0.4.
            'sigma': 2	,  # small imaginary frequency, in 1/T, 2-3.
            'kmax': 15.,		# max wavenumber at w=0, in 1/h, 10-30.
            'pmin': 0.,		# max. phase velocity, in 1/vs, 0 the best.
            'pmax': 1.,		# min. phase velocity, in 1/vs.
            'taper': 0.3,  # for low-pass filter, 0-1.
            'f1': 0,
            'f2': 0,  # for high-pass filter transition band, in Hz.
            'tb': 50	,  # num. of samples before the first arrival.
            'deg2km': 1,
            'flat': False	,  # Earth flattening transformation.
            'r_depth': 0.,  # receiver depth.
            'updn': 0,		# 1=down-going wave only -1=up-going wave only.
            # the input model 3rd column is vp, not vp/vs ratio.
            'kappa': False,

            's_depth': 0,
            'dist': [],
            'model': []
        }

        self.model = model
        self.sdep = sdep
        self.modeltype = modeltype
        self.distance = distance
        self.degrees = degrees
        self.filter = filter
        self.number_of_points = number_of_points
        self.theproperty = theproperty
        self.rdep = rdep
        self.srcType = srcType
        self.updn = updn
        self.dt = dt
        self.smth = smth
        self.dk = dk

        self._getinformation()

    def _getinformation(self):
        # sdep
        self.default_values['s_depth'] = self.sdep
        if(self.sdep < 0):
            raise Exception('sdep>=0')
        # rdep
        self.default_values['r_depth'] = self.rdep
        # modeltype
        if(self.modeltype != None):
            if(self.modeltype == 'f'):
                self.default_values['flat'] = True
                self.default_values['s_depth'] = self.default_values['r0']*np.log(
                    self.default_values['r0']/(self.default_values['r0']-self.default_values['s_depth']))
                self.default_values['r_depth'] = self.default_values['r0']*np.log(
                    self.default_values['r0']/(self.default_values['r0']-self.default_values['r_depth']))
            elif(self.modeltype=='k'):
                self.default_values['kappa'] = True
            else:
                raise Exception('modeltype must be f or k')
        # degrees
        if(self.degrees == True):
            self.default_values['deg2km'] = 6371*3.14159/180.
        elif(self.degrees == False):
            pass
        else:
            raise Exception('degrees must be True or False')
        # filter
        if(self.filter != None):
            if(len(list(self.filter)) == 2):
                self.default_values['f1'] = self.filter[0]
                self.default_values['f2'] = self.filter[1]
            elif(len(list(self.filter)) == 1):
                self.default_values['taper'] = self.filter[0]
            else:
                raise Exception('must provide f1/f2 or taper')
        # initial -N
        self.default_values['nft'] = self.number_of_points
        self.default_values['dt'] = self.dt
        self.default_values['smth'] = self.smth
        self.default_values['dk'] = self.dk
        assert (self.number_of_points == int(self.number_of_points) and self.dt > 0 and self.smth > 0 and self.dk > 0)
        # initial -P
        self.default_values['pmin'] = self.theproperty[0]
        self.default_values['pmax'] = self.theproperty[1]
        self.default_values['kmax'] = self.theproperty[2]
        assert (self.theproperty[0] >= 0 and self.theproperty[1] >= 0 and self.theproperty[2] >= 0)
        # srctype
        chooselist = {'dc': 2, 'sf': 1, 'ep': 0}
        self.default_values['src'] = chooselist[self.srcType]
        if(self.srcType not in chooselist):
            raise Exception('srcType must be dc,sf or ep')
        # updn
        self.default_values['updn'] = self.updn
        if(not (self.updn == 1 or self.updn == -1 or self.updn == 0)):
            raise Exception('updn must be 1,-1 or 0')
        # distance
        if(self.distance != None):
            self.default_values['dist'] = self.distance
        else:
            raise Exception('distance must be provided')
        # model
        self._handle_model()

    def _handle_model(self):
        output = {
            'th': np.zeros(0),
            'vs': np.zeros(0),
            'vp': np.zeros(0),
            'rh': np.zeros(0),
            'qa': np.zeros(0),
            'qb': np.zeros(0),
            'src_layer': 0,
            'rcv_layer': 0,
            'num_layer': 0
        }

        model_values=self.model
        row, column = np.shape(model_values)
        if(column != 6):
            model_values = np.c_[model_values, np.zeros(
                (row, 6-column), dtype=model_values.dtype)]
        r, fl = [np.zeros(row) for i in range(2)]
        r[0] = self.default_values['r0']
        fl[:] = 1.
        for i in range(row):
            r[i] -= model_values[i, 0]
            if(self.default_values['flat']):
                fl[i] = self.default_values['r0']/(r[i]+0.5*model_values[i, 0])
        model_values[:, 0] *= fl
        model_values[:, 1] *= fl
        if(self.default_values['kappa']):
            model_values[:, 2] *= model_values[:, 1]
        else:
            model_values[:, 2] *= fl
        flag=True
        for i in range(row):
            if(column < 4 or model_values[i, 3] > 20.):
                flag=False
                model_values[i, 4] = model_values[i, 3]
                model_values[i, 3] = 0.77+0.32*model_values[i, 2]
        if(column < 5 and flag):
            model_values[:, 4] = 500
        if(column < 6):
            model_values[:, 5] = 2*model_values[:, 4]
        model_values[row-1, 0] = 0.
        self.model_values=model_values

        freeSurf = model_values[0, 0] > 0.
        if(not freeSurf):
            if(np.shape(model_values)[0] < 2):
                freeSurf = True
            else:
                freeSurf = False
        if(freeSurf and (self.default_values['s_depth'] < 0. or self.default_values['r_depth'] < 0.)):
            raise Exception("The source or receivers are located in the air.")
            
        if(self.default_values['s_depth'] < self.default_values['r_depth']):
            output['src_layer'] = self._insert_intf(
                self.default_values['s_depth'])
            output['rcv_layer'] = self._insert_intf(
                self.default_values['r_depth'])
        else:
            output['rcv_layer'] = self._insert_intf(
                self.default_values['r_depth'])
            output['src_layer'] = self._insert_intf(
                self.default_values['s_depth'])
        if(self.model_values[output['src_layer'], 2] != self.model_values[output['src_layer']-1, 2]):
            raise Exception("The source is located at a real interface.")

        output['th'] = self.model_values[:, 0]
        output['vs'] = self.model_values[:, 1]
        output['vp'] = self.model_values[:, 2]
        output['rh'] = self.model_values[:, 3]
        output['qb'] = self.model_values[:, 4]
        output['qa'] = self.model_values[:, 5]
        output['num_layer'] = np.shape(self.model_values)[0]

        self.default_values['model']=output

    def _insert_intf(self, zs):
        n, m = np.shape(self.model_values)
        dep = 0.
        i = 0
        for k in range(n):
            dep += self.model_values[k, 0]
            if(dep > zs or i == n-1):
                break
            i += 1
        intf = i
        if((i > 0 and zs == dep-self.model_values[i, 0]) or (i == 0 and zs == 0)):
            return intf
        dd = dep-zs
        self.model_values = np.concatenate(
            (self.model_values, np.zeros((1, m))), axis=0)
        self.model_values[intf+1:, :] = self.model_values[intf:-1, :]
        self.model_values[intf, 0] -= dd
        if(dd > 0):
            self.model_values[intf+1,0]=dd
        if( self.model_values[0,0]<0.):
            self.default_values['s_depth'] -= self.model_values[0, 0]
            self.default_values['r_depth'] -= self.model_values[0, 0]
            self.model_values[0, 0] = 0.
        return intf+1

    def _runtravel(self):
        self.model=self.default_values['model']
        input={
            'num_lay':self.model['num_layer'],
            'src_lay':self.model['src_layer'],
            'rcv_lay':self.model['rcv_layer'],
            'thk':self.model['th'],
            'vps':self.model['vp'],
            'x':self.default_values['dist']
        }
        vpout=Travel(input).run()
        input['vps']=self.model['vs']
        vsout=Travel(input).run()
        t0=vpout[1]
        sac_com=[{'tp':vpout[1][i],'ts':vsout[1][i]} for i in range(len(list(self.default_values['dist'])))]

        dn=0
        pa=0
        sa=0
        for i in range(len(list(self.default_values['dist']))):
            tp=vpout[1][i]
            if(vpout[2][i]>tp and vpout[3][i]<1/7.): # down going Pn
                pa=self.model['vp'][self.model['src_layer']]*vpout[3][i]
                dn=1
            else:
                pa=self.model['vp'][self.model['src_layer']]*vpout[4][i]
                dn=-1
            sac_com[i]['pa']=np.rad2deg(np.arctan2(pa,dn*np.sqrt(np.abs(1-pa**2))))

            ts=vsout[1][i]
            if(vsout[2][i]>ts and vsout[3][i]<1/4.): # down going Sn
                sa=self.model['vs'][self.model['src_layer']]*vsout[3][i]
                dn=1
            else:
                sa=self.model['vs'][self.model['src_layer']]*vsout[4][i]
                dn=-1
            sac_com[i]['sa']=np.rad2deg(np.arctan2(sa,dn*np.sqrt(np.abs(1-sa**2))))

        return t0,sac_com



        
    def _prepareGFinput(self):
        self.model=self.default_values['model']
        t0,sac_com=self._runtravel()

        self.forGFinput={
            'mb':self.model['num_layer'],
            'src':self.model['src_layer']+1,
            'stype':self.default_values['src'],
            'rcv':self.model['rcv_layer']+1,
            'updn':self.default_values['updn'],
            'd':self.model['th'],
            'a':self.model['vp'],
            'b':self.model['vs'],
            'rho':self.model['rh'],
            'qa':self.model['qa'],
            'qb':self.model['qb'],
            'sigma':2,
            'nfft':self.default_values['nft'],
            'dt':self.default_values['dt'],
            'taper':self.default_values['taper'],
            'tb':self.default_values['tb'],
            'smth':self.default_values['smth'],
            'wc1':int(self.default_values['f1']*self.default_values['nft']*self.default_values['dt'])+1,
            'wc2':int(self.default_values['f2']*self.default_values['nft']*self.default_values['dt'])+1,
            'pmin':self.default_values['pmin'],
            'pmax':self.default_values['pmax'],
            'dk':self.default_values['dk'],
            'kc':self.default_values['kmax'],
            'x':self.default_values['dist'],
            't0':t0,
            'sac_com':sac_com
        }

    def run(self):
        self._prepareGFinput()
        gfresult=MainFK(self.forGFinput).run()
        return gfresult
        