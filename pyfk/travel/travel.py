import numba
import numpy as np
from numba import float64, vectorize

import pyfk.travel.setting as setting
import pyfk.travel.tau_p as tau_p


class Travel_input():
    def __init__(self, input):
        setting.init_travel_taup()

        setting.num_lay = input['num_lay']
        setting.src_lay = input['src_lay']
        setting.rcv_lay = input['rcv_lay']
        setting.thk = input['thk']
        temp = input['vps']
        setting.vps = np.zeros((2, np.shape(temp)[0]))
        setting.vps[0, :] = temp
        setting.x = input['x']
        self.check_input()

    def check_input(self):
        if(setting.src_lay < setting.rcv_lay):
            setting.rcv_lay, setting.src_lay = setting.src_lay, setting.rcv_lay
        setting.vps[0, :] = 1./setting.vps[0, :]**2
        setting.vps[1, :] = setting.vps[0, :]


class Travel(Travel_input):
    def __init__(self, input):
        super(Travel, self).__init__(input)

    def run(self):
        xlist = setting.x
        t0, td = [np.zeros_like(xlist, dtype=np.float) for i in range(2)]
        p0, pd = [np.zeros_like(xlist, dtype=np.complex) for i in range(2)]
        tau_p.runtravel(xlist, t0, td, p0, pd, setting.vps, setting.thk)
        return xlist, t0, td, p0.real, pd.real
