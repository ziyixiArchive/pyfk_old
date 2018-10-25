# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Purpose: python module for F-K.
# Author:
#   Ziyixi      4/5/18      ESS, USTC             Python code author
# 	Lupei Zhu	5/3/96		Seismo Lab, Caltech   Initial Fortran code author
# -----------------------------------------------------------------------------

"""
pyfk: A Python Toolbox for computing 1D synthetic seismic waveform
==================================================================
:copyright:
    L. Zhu (zhul@slu.edu)
    Ziyi Xi (ziyixi@mail.ustc.edu.cn)
:license:
    Apache License, Version 2.0
    (https://www.apache.org/licenses/LICENSE-2.0)
"""


from pyfk.core.fk import FK
from pyfk.core.syn import SYN
__all__=['FK','SYN']
