import sys
import glowpython as glow

def calc_glow_generic(time, nbins, kwargs, *vars):
    try:
        vars, tec, idx = vars
        lat, lon = vars
    except:
        print('Error:', vars)
        return -1, None
    iono = glow.generic(time, lat, lon, nbins, tec=tec, **kwargs)
    return idx, iono