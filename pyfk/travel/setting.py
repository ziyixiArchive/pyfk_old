def init_travel_taup():
    global namelist_taup
    namelist_taup = {'src_lay', 'rcv_lay', 'thk', 'vps', 'x', 'num_lay'}

    for item in namelist_taup:
        exec(item+'=0', globals())
