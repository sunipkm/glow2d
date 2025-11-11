import sys
import glowpython2 as glow

def calc_glow_generic(time, nbins, kwargs, *vars):
    try:
        vars, tec = vars
        lat, lon = vars
    except:
        print('Error:', vars)
        return -1, None
    iono = glow.generic(time, lat, lon, nbins, tec=tec, **kwargs)
    return iono